import pyaudio
import numpy as np
import time
import torch
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

# Load the feature extractor and model for emotion recognition
feature_extractor = AutoFeatureExtractor.from_pretrained("Wiam/wav2vec2-lg-xlsr-en-speech-emotion-recognition-finetuned-ravdess-v8")
model = AutoModelForAudioClassification.from_pretrained("Wiam/wav2vec2-lg-xlsr-en-speech-emotion-recognition-finetuned-ravdess-v8")

# PyAudio parameters
FORMAT = pyaudio.paInt16     # 16-bit integer format
CHANNELS = 1                 # Mono audio
RATE = 16000                 # 16 kHz (model's expected sampling rate)
CHUNK = 1024                 # Number of audio frames per buffer

# Silence detection parameters
THRESHOLD = 3000              # RMS threshold for detecting speech (adjust if needed)
SILENCE_DURATION = 2.0       # Duration (in seconds) of silence to stop recording

p = pyaudio.PyAudio()

# List available input devices (for your reference)
print("Available audio input devices:")
for i in range(p.get_device_count()):
    dev = p.get_device_info_by_index(i)
    if dev["maxInputChannels"] > 0:
        print(f"Device {i}: {dev['name']}")

# Set device_index to the one corresponding to your internal microphone.
# You can change this to the desired device index (e.g., 6 for 'pulse', or 7 for 'default').
device_index = 6

# Open the audio stream for recording
stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True,
                frames_per_buffer=CHUNK, input_device_index=device_index)

print("\nListening... (Speak to start recording, silence for 2 seconds to process)")

def rms(data):
    # Convert byte data to a numpy array of int16 and compute RMS
    samples = np.frombuffer(data, dtype=np.int16)
    if samples.size == 0:
        return 0.0
    return np.sqrt(np.mean(samples.astype(np.float32)**2))

recording = False
frames = []
silence_start = None

try:
    while True:
        data = stream.read(CHUNK, exception_on_overflow=False)
        amplitude = rms(data)
        #print(f"Amplitude: {amplitude}")
        
        # If not recording, check if sound exceeds the threshold to start recording
        if not recording:
            if amplitude > THRESHOLD:
                print("\nSpeech detected, start recording...")
                recording = True
                frames.append(data)
                silence_start = None
        else:
            frames.append(data)
            if amplitude < THRESHOLD:
                if silence_start is None:
                    silence_start = time.time()
                elif time.time() - silence_start > SILENCE_DURATION:
                    print("Silence detected for 2 seconds, stop recording and processing...")
                    
                    # Combine recorded frames into a single byte string
                    audio_data = b"".join(frames)
                    # Convert bytes to numpy array of int16, then to float32 normalized to [-1, 1]
                    waveform = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
                    # Convert numpy array to a torch tensor
                    waveform_tensor = torch.from_numpy(waveform)
                    
                    # Process the waveform with the feature extractor
                    inputs = feature_extractor(waveform_tensor, sampling_rate=RATE, return_tensors="pt", padding=True)
                    
                    # Run emotion classification
                    with torch.no_grad():
                        outputs = model(**inputs)
                        logits = outputs.logits
                    probabilities = torch.nn.functional.softmax(logits, dim=-1)
                    
                    # Get predicted emotion label and scores for each emotion
                    predicted_class_idx = torch.argmax(probabilities, dim=-1).item()
                    predicted_emotion = model.config.id2label[predicted_class_idx]
                    print(f"\nPredicted Emotion: {predicted_emotion}")
                    print("Scores:")
                    scores = probabilities[0].tolist()
                    for idx, label in sorted(model.config.id2label.items()):
                        print(f"  {label}: {scores[idx]:.4f}")
                    
                    # Reset for the next recording
                    recording = False
                    frames = []
                    silence_start = None
            else:
                silence_start = None

except KeyboardInterrupt:
    print("Exiting...")

# Clean up
stream.stop_stream()
stream.close()
p.terminate()
