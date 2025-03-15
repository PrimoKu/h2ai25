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

class BloodPressureAgent:
    def __init__(self, model="gpt-4", temperature=0, api_key=None):
        self.api_key = api_key or config.OPENAI_API_KEY
        self.model = ChatOpenAI(model=model, temperature=temperature, api_key=self.api_key)
        self.file_lock = threading.Lock()  # Lock for thread-safe file operations
        self._last_analysis_time = None    # Track when we last did an analysis
        self._running = False              # Flag to prevent multiple analyses at once
        
        # Define the cardiovascular expert persona
        self.persona = """You are a medical assistant specializing in cardiovascular health with extensive experience in blood pressure monitoring and analysis. 

                            Your responsibilities:
                            1. Carefully analyze blood pressure readings to identify patterns and anomalies
                            2. Interpret systolic and diastolic values according to medical guidelines
                            3. Provide clear insights about cardiovascular health based on the readings
                            4. Detect potential health risks or concerning trends in the data
                            5. Suggest appropriate follow-up actions when necessary

                            When analyzing blood pressure:
                            - Normal range: ~120/80 mmHg
                            - Elevated: Systolic 120-129 and diastolic <80
                            - Hypertension Stage 1: Systolic 130-139 or diastolic 80-89
                            - Hypertension Stage 2: Systolic ≥140 or diastolic ≥90
                            - Hypertensive Crisis: Systolic >180 and/or diastolic >120

                            Always maintain a professional, informative tone while making medical information accessible."""
        
        # Set up the chain with persona
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", self.persona),
            ("human", "{user_prompt}\n\nData: {data}")
        ])
        
        self.chain = self.prompt | self.model | StrOutputParser()
    
    def get_data_from_last_minute(self):
        """Fetches last minute's blood pressure data."""
        if not os.path.exists(config.BLOOD_PRESSURE_CSV):
            return []
        
        one_minute_ago = datetime.now() - timedelta(minutes=1)
        relevant_data = []

        with open(config.BLOOD_PRESSURE_CSV, "r") as f:
            reader = csv.reader(f)
            next(reader)  # Skip header
            for row in reader:
                timestamp = datetime.strptime(row[0], "%Y-%m-%d %H:%M:%S")
                if one_minute_ago <= timestamp <= datetime.now():
                    relevant_data.append({
                        "timestamp": row[0], 
                        "systolic": row[1], 
                        "diastolic": row[2]
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
            if os.path.exists(config.ANALYZED_BLOOD_PRESSURE_JSON):
                with open(config.ANALYZED_BLOOD_PRESSURE_JSON, "r") as f:
                    try:
                        previous_results = json.load(f)
                    except json.JSONDecodeError:
                        pass

            previous_results.append(result_data)

            with open(config.ANALYZED_BLOOD_PRESSURE_JSON, "w") as f:
                json.dump(previous_results, f, indent=4)

            timestamp = result_data['timestamp']
            print(f"[Blood Pressure Agent] Analysis saved at {timestamp}.")
    
    def analyze(self):
        """Runs the blood pressure analysis."""
        # Set running flag to prevent multiple concurrent analyses
        if self._running:
            print("[Blood Pressure Agent] Analysis already in progress, skipping.")
            return
            
        self._running = True
        
        try:
            # Get latest data
            data = self.get_data_from_last_minute()
            if not data:
                print("[Blood Pressure Agent] No data available for analysis.")
                self._running = False
                return
            
            # Create a detailed prompt for analysis
            analysis_prompt = """Please analyze these blood pressure readings. 
            Identify any patterns, abnormal readings, or concerning trends. 
            Consider both systolic and diastolic values and their relationship. 
            Provide a thorough assessment of cardiovascular health based on these readings.
            If appropriate, suggest follow-up actions."""
            
            # Run the analysis through the LLM chain
            analysis = self.chain.invoke({
                "user_prompt": analysis_prompt,
                "data": json.dumps(data, indent=2)
            })
            
            # Save results
            self.save_analysis_results(analysis)
            
            # Update last analysis time
            self._last_analysis_time = datetime.now()
            print(f"[Blood Pressure Agent] Completed analysis.")
            
        except Exception as e:
            print(f"[Blood Pressure Agent] Error during analysis: {str(e)}")
        finally:
            # Always reset running flag when done
            self._running = False
    
    def run(self):
        """Runs the analysis loop with proper timing."""
        print("[Blood Pressure Agent] Started monitoring. Will analyze data every minute.")
        
        while True:
            current_time = datetime.now()
            
            # If it's the first run or at least 1 minute has passed since last analysis
            if (self._last_analysis_time is None or 
                (current_time - self._last_analysis_time).total_seconds() >= 30):
                
                # Start analysis in a separate thread to not block the main loop
                threading.Thread(target=self.analyze, daemon=True).start()
                
            # Sleep for a short time before checking again
            time.sleep(10)  # Check every 10 seconds instead of blocking for a full minute