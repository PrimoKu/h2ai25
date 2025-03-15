import pyaudio
import numpy as np
import time
import torch
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification


class AudioEmotionRecognizer:
    def __init__(self, config):
        self.config = config
        self.p = pyaudio.PyAudio()
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(config["model_name"])
        self.model = AutoModelForAudioClassification.from_pretrained(config["model_name"])
        self.device_index = config["device_index"]
        self.stream = self._initialize_audio_stream()
        self.recording = False
        self.frames = []
        self.silence_start = None

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
                print(f"Amplitude: {amplitude}")
                
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

    def _analyze_audio(self):
        print("Silence detected for 2 seconds, stop recording and processing...")
        audio_data = b"".join(self.frames)
        waveform = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
        waveform_tensor = torch.from_numpy(waveform)
        inputs = self.feature_extractor(waveform_tensor, sampling_rate=self.config["rate"], return_tensors="pt", padding=True)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
        
        predicted_class_idx = torch.argmax(probabilities, dim=-1).item()
        predicted_emotion = self.model.config.id2label[predicted_class_idx]
        print(f"\nPredicted Emotion: {predicted_emotion}")
        print("Scores:")
        scores = probabilities[0].tolist()
        for idx, label in sorted(self.model.config.id2label.items()):
            print(f"  {label}: {scores[idx]:.4f}")

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
        "threshold": 1000,
        "silence_duration": 2.0,
        "device_index": 7,  # Change as needed
    }
    
    recognizer = AudioEmotionRecognizer(config)
    recognizer.list_audio_devices()
    recognizer.process_audio()

