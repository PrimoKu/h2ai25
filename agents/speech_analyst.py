import json
import csv
import threading
import os
import schedule
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
        """Saves analysis results to a JSON file."""
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
        
        print("[Speech Analyst Agent] Analysis saved.")
    
    def analyze(self):
        """Runs the speech analysis."""
        try:
            # Retrieve latest speech data
            data = self.get_data_from_last_minute()
            if not data:
                print("[Speech Analyst Agent] No data available for analysis.")
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
            
            print("[Speech Analyst Agent] Completed analysis.")
            
        except Exception as e:
            print(f"[Speech Analyst Agent] Error during analysis: {str(e)}")
    
    def run(self):
        """Schedules speech analysis every minute."""
        schedule.every(1).minutes.do(lambda: threading.Thread(target=self.analyze).start())
        
        print("[Speech Analyst Agent] Started monitoring. Will analyze data every minute.")
        while True:
            schedule.run_pending()
            time.sleep(1)

if __name__ == "__main__":
    agent = SpeechAnalystAgent()
    agent.run()
