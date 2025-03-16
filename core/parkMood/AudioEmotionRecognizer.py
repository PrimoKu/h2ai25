import pyaudio
import numpy as np
import time
import torch
import os
import csv
import datetime
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
from scipy.signal import butter, filtfilt  # for filtering

class AudioEmotionRecognizer:
    def __init__(self, config):
        self.config = config
        self.p = pyaudio.PyAudio()
        import os
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(config["model_name"])
        self.model = AutoModelForAudioClassification.from_pretrained(config["model_name"])
        self.device_index = config["device_index"]
        self.stream = self._initialize_audio_stream()
        self.recording = False
        self.frames = []
        self.silence_start = None
        self.csv_path = config["csv_path"]
        
        # Ensure the directory exists
        os.makedirs(os.path.dirname(self.csv_path), exist_ok=True)
        
        # Check if the CSV file exists, if not create it with headers
        self._initialize_csv()

    def _initialize_csv(self):
        """Initialize the CSV file with headers if it doesn't exist"""
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([
                    'timestamp', 
                    'Fundamental Frequency(Hz)', 
                    'Jitter(%)', 
                    'Shimmer(%)', 
                    'Volumn(RMS)', 
                    'Volumn Status',
                    'neutral(%)',
                    'calm(%)',
                    'happy(%)',
                    'sad(%)',
                    'angry(%)',
                    'fearful(%)', 
                    'disgust(%)',
                    'surprised(%)'
                ])
            print(f"Created new CSV file at {self.csv_path}")

    def _initialize_audio_stream(self):
        return self.p.open(
            format=self.config["audio_format"],
            channels=self.config["channels"],
            rate=self.config["rate"],
            input=True,
            frames_per_buffer=self.config["chunk"],
            input_device_index=self.device_index,
        )

    def _rms(self, data):
        samples = np.frombuffer(data, dtype=np.int16)
        return np.sqrt(np.mean(samples.astype(np.float32) ** 2)) if samples.size else 0.0

    def list_audio_devices(self):
        print("Available audio input devices:")
        for i in range(self.p.get_device_count()):
            dev = self.p.get_device_info_by_index(i)
            if dev["maxInputChannels"] > 0:
                print(f"Device {i}: {dev['name']}")

    def process_audio(self):
        print("\nListening... (Speak to start recording, silence for 2 seconds to process)")
        try:
            while True:
                data = self.stream.read(self.config["chunk"], exception_on_overflow=False)
                amplitude = self._rms(data)
                print(f"Amplitude: {amplitude:.2f}")
                
                if not self.recording and amplitude > self.config["threshold"]:
                    print("\nSpeech detected, start recording...")
                    self.recording = True
                    self.frames.append(data)
                    self.silence_start = None
                elif self.recording:
                    self.frames.append(data)
                    if amplitude < self.config["threshold"]:
                        if self.silence_start is None:
                            self.silence_start = time.time()
                        elif time.time() - self.silence_start > self.config["silence_duration"]:
                            self._analyze_audio()
                            self._reset()
                    else:
                        self.silence_start = None
        except KeyboardInterrupt:
            print("Exiting...")
            self.cleanup()

    def _bandpass_filter(self, signal, lowcut, highcut, fs, order=3):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        return filtfilt(b, a, signal)

    def _compute_pitch(self, frame, fs):
        # Remove DC offset
        frame = frame - np.mean(frame)
        # Compute autocorrelation
        corr = np.correlate(frame, frame, mode='full')
        corr = corr[len(corr)//2:]
        # Define plausible pitch lag range for human speech (80Hz to 400Hz)
        min_lag = int(fs / 400)
        max_lag = int(fs / 80)
        if max_lag >= len(corr):
            max_lag = len(corr) - 1
        segment = corr[min_lag:max_lag]
        if len(segment) == 0:
            return None
        peak_index = np.argmax(segment) + min_lag
        # Ensure the peak is significant enough relative to the zero lag
        if corr[peak_index] < 0.3 * corr[0]:
            return None
        pitch = fs / peak_index
        return pitch

    def _analyze_voice_features(self, waveform):
        fs = self.config["rate"]
        # Apply bandpass filtering to reduce noise effects
        filtered_waveform = self._bandpass_filter(waveform, lowcut=80, highcut=400, fs=fs, order=3)
        # Define frame parameters (30ms window, 10ms hop)
        frame_size = int(0.03 * fs)
        hop_size = int(0.01 * fs)
        pitches = []
        amplitudes = []
        energy_threshold = 0.01  # threshold to skip silent frames
        for start in range(0, len(filtered_waveform) - frame_size, hop_size):
            frame = filtered_waveform[start:start + frame_size]
            rms_frame = np.sqrt(np.mean(frame ** 2))
            if rms_frame < energy_threshold:
                continue  # skip low-energy (possibly unvoiced) frames
            pitch = self._compute_pitch(frame, fs)
            if pitch is not None:
                pitches.append(pitch)
                amplitudes.append(rms_frame)
        # Calculate overall features if enough voiced frames were detected
        if len(pitches) < 2 or len(amplitudes) < 2:
            Fo = None
            jitter = None
            shimmer = None
        else:
            Fo = np.mean(pitches)
            # Convert pitch (Hz) to period (seconds)
            periods = [1.0 / p for p in pitches if p > 0]
            if len(periods) >= 2:
                period_diffs = np.abs(np.diff(periods))
                jitter = 100 * np.mean(period_diffs) / np.mean(periods)
            else:
                jitter = None
            amp_diffs = np.abs(np.diff(amplitudes))
            shimmer = 100 * np.mean(amp_diffs) / np.mean(amplitudes)
        # Here, we use the RMS of the filtered waveform as a proxy for volume.
        # For speakers with Parkinson's, reduced volume (hypophonia) is of interest.
        volume = np.sqrt(np.mean(filtered_waveform ** 2))
        # Compare volume against a threshold to flag reduced volume
        reduced_volume = volume < self.config.get("volume_threshold", 0.02)
        volume_status = "Reduced" if reduced_volume else "Normal"
        
        return {
            "Fo": Fo, 
            "Jitter": jitter, 
            "Shimmer": shimmer, 
            "Volume": volume, 
            "VolumeStatus": volume_status
        }

    def _analyze_audio(self):
        print("\nSilence detected for 2 seconds, stop recording and processing...")
        audio_data = b"".join(self.frames)
        waveform = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0

        # Analyze voice features first
        voice_features = self._analyze_voice_features(waveform)
        print("\n--- Voice Features ---")
        if voice_features["Fo"] is not None:
            print(f"Fundamental Frequency (Fo): {voice_features['Fo']:.2f} Hz")
        else:
            print("Fundamental Frequency (Fo): N/A")
        if voice_features["Jitter"] is not None:
            print(f"Jitter: {voice_features['Jitter']:.2f} %")
        else:
            print("Jitter: N/A")
        if voice_features["Shimmer"] is not None:
            print(f"Shimmer: {voice_features['Shimmer']:.2f} %")
        else:
            print("Shimmer: N/A")
        print(f"Volume (RMS): {voice_features['Volume']:.4f}")
        print(f"Volume Status: {voice_features['VolumeStatus']}")

        # Then run mood/emotion detection
        waveform_tensor = torch.from_numpy(waveform)
        inputs = self.feature_extractor(waveform_tensor, sampling_rate=self.config["rate"],
                                        return_tensors="pt", padding=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
        predicted_class_idx = torch.argmax(probabilities, dim=-1).item()
        predicted_emotion = self.model.config.id2label[predicted_class_idx]
        
        print("\n--- Mood Detection ---")
        print(f"Predicted Emotion: {predicted_emotion}")
        print("Scores:")
        
        # Get all emotion scores
        scores = probabilities[0].tolist()
        emotion_scores = {}
        for idx, label in sorted(self.model.config.id2label.items()):
            # Store raw probability scores (0-1)
            emotion_scores[label] = scores[idx]
            # Display as percentage for user-friendly output
            print(f"  {label}: {scores[idx] * 100:.2f}%")
            
        # Save data to CSV
        self._save_to_csv(voice_features, emotion_scores)

    def _save_to_csv(self, voice_features, emotion_scores):
        """Save the analysis results to CSV file"""
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Normalize Jitter and Shimmer from percentage (0-100) to 0-1 range
        jitter_normalized = voice_features["Jitter"] / 100 if voice_features["Jitter"] is not None else ""
        shimmer_normalized = voice_features["Shimmer"] / 100 if voice_features["Shimmer"] is not None else ""
        
        # Normalize emotion scores from percentage (0-100) to 0-1 range
        normalized_emotions = {emotion: score for emotion, score in emotion_scores.items()}
        
        # Prepare the row data
        row_data = [
            timestamp,
            voice_features["Fo"] if voice_features["Fo"] is not None else "",
            jitter_normalized,
            shimmer_normalized,
            voice_features["Volume"],  # Already in 0-1 range
            voice_features["VolumeStatus"],
            normalized_emotions.get("neutral", 0),
            normalized_emotions.get("calm", 0),
            normalized_emotions.get("happy", 0),
            normalized_emotions.get("sad", 0),
            normalized_emotions.get("angry", 0),
            normalized_emotions.get("fearful", 0),
            normalized_emotions.get("disgust", 0),
            normalized_emotions.get("surprised", 0)
        ]
        
        # Append to CSV
        try:
            with open(self.csv_path, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(row_data)
            print(f"\nResults saved to {self.csv_path}")
        except Exception as e:
            print(f"Error saving to CSV: {e}")

    def _reset(self):
        self.recording = False
        self.frames = []
        self.silence_start = None

    def cleanup(self):
        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()


if __name__ == "__main__":
    config = {
        "model_name": "Wiam/wav2vec2-lg-xlsr-en-speech-emotion-recognition-finetuned-ravdess-v8",
        "audio_format": pyaudio.paInt16,
        "channels": 1,
        "rate": 16000,
        "chunk": 1024,
        "threshold": 400,         # amplitude threshold for speech detection
        "silence_duration": 2.0,
        "volume_threshold": 0.02,  # RMS threshold below which volume is considered reduced
        "device_index": None,      # Change as needed
        "csv_path": "./web_app/data_storage/speech.csv"  # Path to save CSV data
    }
    
    recognizer = AudioEmotionRecognizer(config)
    recognizer.list_audio_devices()
    recognizer.process_audio()
