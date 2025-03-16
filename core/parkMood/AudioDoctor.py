import socket
import time
import speech_recognition as sr
import pyttsx3
import requests
import json
from vosk import Model, KaldiRecognizer
import pyaudio
import threading

class HotwordRecognizer:
    def __init__(self, hotwords=None, send_port=65432, feedback_port=65433, 
                 api_url="http://127.0.0.1:5000/api/message", host="localhost"):
        if hotwords is None:
            hotwords = ["ready"]
        self.hotwords = [hw.lower() for hw in hotwords]
        self.send_port = send_port
        self.feedback_port = feedback_port
        self.api_url = api_url
        self.host = host

        # Improved audio configuration
        self.recognizer = sr.Recognizer()
        self.recognizer.pause_threshold = 1.5  # Optimal pause detection
        self.recognizer.energy_threshold = 4000  # Initial energy threshold
        self.recognizer.dynamic_energy_threshold = True
        self.recognizer.dynamic_energy_adjustment_damping = 0.15
        self.recognizer.dynamic_energy_ratio = 1.5

        # VOSK offline model setup
        self.vosk_model = Model(lang="en-us")
        self.vosk_recognizer = KaldiRecognizer(self.vosk_model, 16000)
        self.audio_interface = pyaudio.PyAudio()
        
        # Audio configuration parameters
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.RATE = 16000
        self.CHUNK = 1024

        self.microphone = sr.Microphone()
        self.tts_engine = pyttsx3.init()
        self.tts_engine.setProperty('rate', 150)
        self.tts_engine.setProperty('volume', 1.0)

        # Audio stream for VOSK
        self.vosk_stream = None

    def speak(self, text):
        """Text-to-speech method to speak the given text"""
        if not text:
            return
        try:
            print(f"Speaking: {text}")
            self.tts_engine.say(text)
            self.tts_engine.runAndWait()
        except Exception as e:
            print(f"TTS error: {e}")

    def robust_listen(self, timeout=5, phrase_time_limit=10):
        """Hybrid listening using both Google and VOSK with retry mechanism"""
        with self.microphone as source:
            print("Calibrating microphone...")
            self.recognizer.adjust_for_ambient_noise(source, duration=2)
            
            try:
                audio = self.recognizer.listen(
                    source, 
                    timeout=timeout,
                    phrase_time_limit=phrase_time_limit
                )
            except sr.WaitTimeoutError:
                return ""

            # Try Google first
            try:
                text = self.recognizer.recognize_google(audio)
                print(f"Google recognition: {text}")
                return text.lower()
            except (sr.UnknownValueError, sr.RequestError):
                pass

            # Fallback to VOSK
            try:
                data = audio.get_raw_data()
                self.vosk_recognizer.AcceptWaveform(data)
                result = json.loads(self.vosk_recognizer.Result())
                text = result.get('text', '')
                print(f"VOSK recognition: {text}")
                return text.lower()
            except Exception as e:
                print(f"VOSK error: {e}")
                return ""

    def continuous_listen(self, stop_event):
        """Continuous listening using VOSK for real-time processing"""
        self.vosk_stream = self.audio_interface.open(
            format=self.FORMAT,
            channels=self.CHANNELS,
            rate=self.RATE,
            input=True,
            frames_per_buffer=self.CHUNK
        )

        print("Starting continuous listening...")
        while not stop_event.is_set():
            data = self.vosk_stream.read(self.CHUNK, exception_on_overflow=False)
            if self.vosk_recognizer.AcceptWaveform(data):
                result = json.loads(self.vosk_recognizer.Result())
                text = result.get('text', '').lower()
                if text:
                    print(f"Continuous recognition: {text}")
                    yield text
                    
    def send_api_message(self, text, role):
        """Send message to API endpoint"""
        if not text:
            return False
            
        payload = {
            "speaker": role.lower(),
            "context": text.strip()
        }
        
        try:
            response = requests.post(
                self.api_url,
                json=payload,
                headers={'Content-Type': 'application/json'},
                timeout=3
            )
            if response.status_code == 200:
                print(f"API success: {text}")
                return True
            print(f"API error: {response.status_code}")
            return False
        except requests.exceptions.RequestException as e:
            print(f"API connection failed: {e}")
            return False
        except Exception as e:
            print(f"API error: {e}")
            return False
    
    def send_text(self, text):
        """Improved text sending with retry mechanism"""
        if not text:
            return False
            
        for attempt in range(3):
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                    sock.settimeout(2)
                    sock.connect((self.host, self.send_port))
                    sock.sendall(text.encode('utf-8'))
                print(f"Sent text to {self.host}:{self.send_port}")
                return True
            except Exception as e:
                print(f"Send error (attempt {attempt+1}): {e}")
                time.sleep(0.5)
        return False

    def receive_feedback(self):
        """Improved feedback reception with timeout handling"""
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_sock:
                server_sock.settimeout(10)
                server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                server_sock.bind((self.host, self.feedback_port))
                server_sock.listen(1)
                print(f"Waiting for feedback (max 10s)...")
                conn, addr = server_sock.accept()
                with conn:
                    conn.settimeout(5)
                    data = conn.recv(1024)
                    return data.decode('utf-8')
        except socket.timeout:
            print("Feedback timeout")
            return ""
        except Exception as e:
            print(f"Feedback error: {e}")
            return ""

    def stream_mode(self, initial_text):
        """Improved streaming mode with continuous listening"""
        if not self.send_text(initial_text):
            print("Failed to send initial text")
            return
            
        feedback = self.receive_feedback()
        if feedback:
            self.send_api_message(feedback, "agent")
            self.speak(feedback)
        else:
            print("No initial feedback")

        # Use event flag to control continuous listening
        stop_event = threading.Event()
        
        try:
            for text in self.continuous_listen(stop_event):
                if not text:
                    continue
                    
                self.send_text(text)
                self.send_api_message(text, "user")
                
                feedback = self.receive_feedback()
                if feedback:
                    self.send_api_message(feedback, "agent")
                    self.speak(feedback)
                else:
                    print("No command feedback")
                    
                # Check for exit command
                if "exit" in text or "stop" in text or "quit" in text:
                    print("Exit command detected")
                    stop_event.set()
                    break
        except Exception as e:
            print(f"Stream mode error: {e}")
        finally:
            stop_event.set()
            if self.vosk_stream:
                self.vosk_stream.stop_stream()
            print("Exiting streaming mode")

    def run(self):
        """Main loop with improved reliability"""
        print("Starting robust hotword detection")
        try:
            while True:
                text = self.robust_listen(timeout=7, phrase_time_limit=8)
                if text:
                    self.send_api_message(text, "user")
                    if any(hw in text for hw in self.hotwords):
                        print(f"Hotword detected: {text}")
                        self.stream_mode(text)
                    else:
                        print("No hotword detected")
                else:
                    print("No speech detected - adjusting thresholds")
                    with self.microphone as source:
                        self.recognizer.adjust_for_ambient_noise(source, duration=1)
        except KeyboardInterrupt:
            print("Keyboard interrupt detected")
        except Exception as e:
            print(f"Run error: {e}")
        finally:
            self.cleanup()
            
    def cleanup(self):
        """Clean up resources properly"""
        print("Cleaning up resources...")
        if self.vosk_stream:
            try:
                self.vosk_stream.stop_stream()
                self.vosk_stream.close()
            except:
                pass
        try:
            self.audio_interface.terminate()
        except:
            pass
        print("Cleanup complete")

if __name__ == "__main__":
    recognizer = HotwordRecognizer()
    recognizer.run()