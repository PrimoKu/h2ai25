import json
import csv
import threading
import os
import time
from datetime import datetime, timedelta
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
import config

class SpeechAnalystAgent:
    def __init__(self, model="gpt-4", temperature=0, api_key=None):
        self.api_key = api_key or config.OPENAI_API_KEY
        self.model = ChatOpenAI(model=model, temperature=temperature, api_key=self.api_key)
        self.file_lock = threading.Lock()  # Lock for thread-safe file operations
        self._last_analysis_time = None    # Track when we last did an analysis
        self._running = False              # Flag to prevent multiple analyses at once
        
        # Define the speech analyst persona
        self.persona = """You are a specialized speech analyst and vocal health expert with extensive experience in analyzing voice features and detecting emotional cues in speech.

                            Your responsibilities:
                            1. Analyze voice features such as Fundamental Frequency (Fo), Jitter, Shimmer, and Volume (RMS) to assess voice quality.
                            2. Evaluate volume status and detect indications of vocal issues, such as reduced volume (possible hypophonia).
                            3. Interpret mood detection results by analyzing the predicted emotion and its associated scores.
                            4. Provide insights on voice quality, potential vocal disorders, and the emotional state of the speaker.
                            5. Offer evidence-based recommendations for follow-up actions when concerning patterns are identified.

                            When analyzing speech data, consider:
                            - Normal voice parameters versus deviations that may suggest vocal strain or pathology.
                            - The interplay between voice features and emotional expression.
                            - Clear, professional communication that makes technical findings accessible.
                            """
        # Set up the prompt chain with the defined persona
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", self.persona),
            ("human", "{user_prompt}\n\nData: {data}")
        ])
        
        self.chain = self.prompt | self.model | StrOutputParser()
    
    def get_data_from_last_minute(self):
        """Fetches last minute's speech data from CSV."""
        if not os.path.exists(config.SPEECH_DATA_CSV):
            return []
        
        one_minute_ago = datetime.now() - timedelta(minutes=1)
        relevant_data = []
        
        # Assumes the CSV has a header row with at least a "timestamp" field.
        with open(config.SPEECH_DATA_CSV, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    timestamp = datetime.strptime(row["timestamp"], "%Y-%m-%d %H:%M:%S")
                except Exception as e:
                    continue  # Skip rows with invalid timestamp formats
                if one_minute_ago <= timestamp <= datetime.now():
                    relevant_data.append(row)
        
        return relevant_data
    
    def save_analysis_results(self, analysis):
        """Saves analysis results to a JSON file with thread safety."""
        with self.file_lock:  # Ensure only one thread writes at a time
            result_data = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "analysis": analysis
            }
            
            previous_results = []
            if os.path.exists(config.ANALYZED_SPEECH_JSON):
                with open(config.ANALYZED_SPEECH_JSON, "r") as f:
                    try:
                        previous_results = json.load(f)
                    except json.JSONDecodeError:
                        pass
            
            previous_results.append(result_data)
            
            with open(config.ANALYZED_SPEECH_JSON, "w") as f:
                json.dump(previous_results, f, indent=4)
            
            timestamp = result_data['timestamp']
            print(f"[Speech Analyst Agent] Analysis saved at {timestamp}.")
    
    def analyze(self):
        """Runs the speech analysis."""
        # Set running flag to prevent multiple concurrent analyses
        if self._running:
            print("[Speech Analyst Agent] Analysis already in progress, skipping.")
            return
            
        self._running = True
        
        try:
            # Retrieve latest speech data
            data = self.get_data_from_last_minute()
            if not data:
                print("[Speech Analyst Agent] No data available for analysis.")
                self._running = False
                return
            
            # Detailed analysis prompt for speech data
            analysis_prompt = """Please analyze the following speech data.
            Examine the voice features such as Fundamental Frequency (Fo), Jitter, Shimmer, and Volume (RMS).
            Evaluate the Volume Status for indications of reduced volume (possible hypophonia).
            Additionally, assess the mood detection results by reviewing the predicted emotion and associated scores.
            Provide insights into any anomalies, potential vocal health issues, or emotional cues, and suggest any follow-up actions if necessary."""
            
            # Run the analysis using the LLM chain
            analysis = self.chain.invoke({
                "user_prompt": analysis_prompt,
                "data": json.dumps(data, indent=2)
            })
            
            # Save the analysis results
            self.save_analysis_results(analysis)
            
            # Update last analysis time
            self._last_analysis_time = datetime.now()
            print(f"[Speech Analyst Agent] Completed analysis.")
            
        except Exception as e:
            print(f"[Speech Analyst Agent] Error during analysis: {str(e)}")
        finally:
            # Always reset running flag when done
            self._running = False
    
    def run(self):
        """Runs the analysis loop with proper timing."""
        print("[Speech Analyst Agent] Started monitoring. Will analyze data every minute.")
        
        while True:
            current_time = datetime.now()
            
            # If it's the first run or at least 1 minute has passed since last analysis
            if (self._last_analysis_time is None or 
                (current_time - self._last_analysis_time).total_seconds() >= 60):
                
                # Start analysis in a separate thread to not block the main loop
                threading.Thread(target=self.analyze, daemon=True).start()
                
            # Sleep for a short time before checking again
            time.sleep(10)  # Check every 10 seconds instead of blocking for a full minute