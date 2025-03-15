import speech_recognition as sr
import pyttsx3

class HotwordRecognizer:
    def __init__(self, hotwords=None, response="Hi I'm here, how can I help you?"):
        if hotwords is None:
            hotwords = ["hey doctor", "hi doctor", "doctor"]
        self.hotwords = [hw.lower() for hw in hotwords]
        self.response = response
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()  # Uses the default microphone
        self.tts_engine = pyttsx3.init()
        
    def speak(self, text):
        """Use text-to-speech to say the provided text."""
        self.tts_engine.say(text)
        self.tts_engine.runAndWait()
    
    def listen_for_hotword(self):
        """Listen for a short phrase and return recognized text in lowercase."""
        with self.microphone as source:
            print("Listening for hotword...")
            # Adjust for ambient noise briefly
            self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
            audio = self.recognizer.listen(source, phrase_time_limit=3)
        try:
            text = self.recognizer.recognize_google(audio)
            print("Heard: " + text)
            return text.lower()
        except sr.UnknownValueError:
            # Could not understand the audio
            return ""
        except sr.RequestError as e:
            print("Request error from Google Speech Recognition service: {0}".format(e))
            return ""
    
    def listen_for_command(self):
        """Listen for a command after the hotword has been triggered."""
        with self.microphone as source:
            print("Listening for command...")
            self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
            audio = self.recognizer.listen(source, phrase_time_limit=5)
        try:
            command = self.recognizer.recognize_google(audio)
            return command
        except sr.UnknownValueError:
            return "Sorry, I did not understand that."
        except sr.RequestError as e:
            return f"Error: {e}"
    
    def run(self):
        """Continuously listen for the hotword, then process the command."""
        print("Hotword recognizer started. Say one of the hotwords to activate.")
        while True:
            text = self.listen_for_hotword()
            if any(hw in text for hw in self.hotwords):
                print("Hotword detected!")
                self.speak(self.response)
                command = self.listen_for_command()
                print("Command received:", command)

if __name__ == "__main__":
    recognizer = HotwordRecognizer()
    recognizer.run()
