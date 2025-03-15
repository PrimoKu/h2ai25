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

class HeartRateAgent:
    def __init__(self, model="gpt-4", temperature=0, api_key=None):
        self.api_key = api_key or config.OPENAI_API_KEY
        self.model = ChatOpenAI(model=model, temperature=temperature, api_key=self.api_key)
        self.file_lock = threading.Lock()  # Lock for thread-safe file operations
        self._last_analysis_time = None    # Track when we last did an analysis
        self._running = False              # Flag to prevent multiple analyses at once

        # Define the heart health expert persona
        self.persona = """You are a specialized cardiologist and heart health expert with extensive experience in heart rate monitoring and analysis.

                            Your responsibilities:
                            1. Carefully analyze heart rate data to identify patterns, trends, and anomalies
                            2. Detect potential arrhythmias, tachycardia, bradycardia, or other cardiac irregularities
                            3. Assess the overall heart health based on heart rate variability and patterns
                            4. Provide evidence-based insights about cardiovascular function
                            5. Suggest appropriate follow-up actions when concerning patterns are detected

                            When analyzing heart rate:
                            - Normal resting heart rate for adults: 60-100 BPM
                            - Athletes may have lower resting rates: 40-60 BPM
                            - Bradycardia: Heart rate below 60 BPM (may be normal for athletes)
                            - Tachycardia: Heart rate above 100 BPM
                            - Heart rate variability (HRV) is important to assess overall cardiac health
                            - Sudden changes in heart rate patterns may indicate stress or underlying conditions

                            Always communicate in a clear, professional manner while making medical information accessible."""
        
        # Set up the chain with persona
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", self.persona),
            ("human", "{user_prompt}\n\nData: {data}")
        ])
        
        self.chain = self.prompt | self.model | StrOutputParser()
    
    def get_data_from_last_minute(self):
        """Fetches last minute's heart rate data."""
        if not os.path.exists(config.HEART_RATE_CSV):
            return []
        
        one_minute_ago = datetime.now() - timedelta(minutes=1)
        relevant_data = []

        with open(config.HEART_RATE_CSV, "r") as f:
            reader = csv.reader(f)
            next(reader)  # Skip header
            for row in reader:
                timestamp = datetime.strptime(row[0], "%Y-%m-%d %H:%M:%S")
                if one_minute_ago <= timestamp <= datetime.now():
                    relevant_data.append({
                        "timestamp": row[0], 
                        "heart_rate": row[1]
                    })

        return relevant_data
    
    def save_analysis_results(self, analysis):
        """Saves analysis results to JSON file with thread safety."""
        with self.file_lock:  # Ensure only one thread writes at a time
            result_data = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "analysis": analysis
            }

            previous_results = []
            if os.path.exists(config.ANALYZED_HEART_RATE_JSON):
                with open(config.ANALYZED_HEART_RATE_JSON, "r") as f:
                    try:
                        previous_results = json.load(f)
                    except json.JSONDecodeError:
                        pass

            previous_results.append(result_data)

            with open(config.ANALYZED_HEART_RATE_JSON, "w") as f:
                json.dump(previous_results, f, indent=4)

            timestamp = result_data['timestamp']
            print(f"[Heart Rate Agent] Analysis saved at {timestamp}.")
    
    def analyze(self):
        """Runs the heart rate analysis."""
        # Set running flag to prevent multiple concurrent analyses
        if self._running:
            print("[Heart Rate Agent] Analysis already in progress, skipping.")
            return
            
        self._running = True
        
        try:
            # Get latest data
            data = self.get_data_from_last_minute()
            if not data:
                print("[Heart Rate Agent] No data available for analysis.")
                self._running = False
                return
            
            # Create a detailed prompt for analysis
            analysis_prompt = """Please analyze these heart rate readings.
            Look for patterns and variations in the heart rate values.
            Identify any signs of tachycardia, bradycardia, or irregular rhythms.
            Assess heart rate variability and its implications for cardiovascular health.
            If concerning patterns are detected, suggest appropriate follow-up actions."""
            
            # Run the analysis through the LLM chain
            analysis = self.chain.invoke({
                "user_prompt": analysis_prompt,
                "data": json.dumps(data, indent=2)
            })
            
            # Save results
            self.save_analysis_results(analysis)
            
            # Update last analysis time
            self._last_analysis_time = datetime.now()
            print(f"[Heart Rate Agent] Completed analysis.")
            
        except Exception as e:
            print(f"[Heart Rate Agent] Error during analysis: {str(e)}")
        finally:
            # Always reset running flag when done
            self._running = False
    
    def run(self):
        """Runs the analysis loop with proper timing."""
        print("[Heart Rate Agent] Started monitoring. Will analyze data every minute.")
        
        while True:
            current_time = datetime.now()
            
            # If it's the first run or at least 1 minute has passed since last analysis
            if (self._last_analysis_time is None or 
                (current_time - self._last_analysis_time).total_seconds() >= 30):
                
                # Start analysis in a separate thread to not block the main loop
                threading.Thread(target=self.analyze, daemon=True).start()
                
            # Sleep for a short time before checking again
            time.sleep(10)  # Check every 10 seconds instead of blocking for a full minute