import torch
import torchaudio
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

# Load the feature extractor and model
feature_extractor = AutoFeatureExtractor.from_pretrained("Wiam/wav2vec2-lg-xlsr-en-speech-emotion-recognition-finetuned-ravdess-v8")
model = AutoModelForAudioClassification.from_pretrained("Wiam/wav2vec2-lg-xlsr-en-speech-emotion-recognition-finetuned-ravdess-v8")

# Load the audio file
waveform, sample_rate = torchaudio.load("test1.wav")  # Replace with your file path

# Convert stereo to mono (if necessary) or remove extra dimension for mono
if waveform.shape[0] > 1:
    waveform = waveform.mean(dim=0)
else:
    waveform = waveform.squeeze(0)  # waveform shape becomes (samples,)

# Resample the audio if necessary
target_sample_rate = 16000
if sample_rate != target_sample_rate:
    resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sample_rate)
    waveform = resampler(waveform)
    sample_rate = target_sample_rate

# Preprocess the audio file
inputs = feature_extractor(waveform, sampling_rate=sample_rate, return_tensors="pt", padding=True)

# Perform emotion classification (prediction)
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits

# Apply softmax to convert logits to probabilities
probabilities = torch.nn.functional.softmax(logits, dim=-1)

# Get the predicted emotion label (if needed)
predicted_class_idx = torch.argmax(probabilities, dim=-1).item()
predicted_emotion = model.config.id2label[predicted_class_idx]

print(f"Predicted Emotion: {predicted_emotion}")

# Output scores for all emotions
print("\nScores for each emotion:")
scores = probabilities[0].tolist()
for idx, label in sorted(model.config.id2label.items()):
    print(f"{label}: {scores[idx]:.4f}")
