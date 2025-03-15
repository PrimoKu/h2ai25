# Audio Emotion Recognition Part

## Installation Guide
### 1. Prerequisites
Ensure you have the following installed:
- **Python 3.8+**
- **pip** (Python package manager)
- **Homebrew (for macOS)**
- **Git LFS (for large model files)**

### 2. Install Dependencies
#### On macOS (using VS Code Terminal)
```sh
# Install Homebrew if not installed
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Git LFS (for large model files)
brew install git-lfs

# Initialize LFS
git lfs install
```

#### 3. Clone the Repository
```sh
git clone git@github.com:PrimoKu/h2ai25.git -b speech_emotion
cd h2ai25
```

#### 4. Create and Activate Virtual Environment
```sh
# Create a virtual environment
python3 -m venv venv

# Activate the virtual environment (macOS/Linux)
source venv/bin/activate
```

#### 5. Install Required Packages
```sh
pip install --upgrade pip
pip install -r requirements.txt
```

---

## Running the Application
```sh
    recognizer = AudioEmotionRecognizer(config)
    recognizer.list_audio_devices()
    recognizer.process_audio()

---

## Configurations
```python
config = {
    "model_name": "Wiam/wav2vec2-lg-xlsr-en-speech-emotion-recognition-finetuned-ravdess-v8",
    "audio_format": pyaudio.paInt16,
    "channels": 1,
    "rate": 16000,
    "chunk": 1024,
    "threshold": 1000,
    "silence_duration": 2.0,
    "device_index": 7,  # Change this to match your microphone device
}
```

---


