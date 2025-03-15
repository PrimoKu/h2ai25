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

class MotorSkillAgent:
    def __init__(self, model="gpt-4", temperature=0, api_key=None):
        self.api_key = api_key or config.OPENAI_API_KEY
        self.model = ChatOpenAI(model=model, temperature=temperature, api_key=self.api_key)
        self.file_lock = threading.Lock()  # Lock for thread-safe file operations
        self._last_analysis_time = None    # Track when we last did an analysis
        self._running = False              # Flag to prevent multiple analyses at once

        # Define the motor skill expert persona with Parkinson's disease expertise
        self.persona = """You are an expert neurologist specializing in motor function and movement disorders, with particular expertise in Parkinson's disease.

                            Your responsibilities:
                            1. Analyze grip strength and coordination measurements to detect patterns, tremors, or motor impairments
                            2. Identify signs of neurodegenerative disorders, particularly Parkinson's disease
                            3. Assess the progression or stability of motor symptoms over time
                            4. Differentiate between normal fatigue-related changes and pathological motor decline
                            5. Provide evidence-based insights about neurological function

                            When analyzing motor skills:
                            - Declining grip strength may indicate neuromuscular weakness or Parkinson's progression
                            - Coordination fluctuations could reveal tremors or dyskinesia
                            - Daily patterns of motor function are important (often better in morning, worse later in day)
                            - Motor symptoms in Parkinson's are often asymmetrical (one side affected more than other)
                            - Medication timing significantly impacts motor performance in Parkinson's patients
                            - "On/off" phenomena in motor function are characteristic of advancing Parkinson's disease

                            Always consider both the absolute values and the trends over time, as progressive decline is more concerning than stable readings, even if below normal thresholds."""
        
        # Set up the chain with persona
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", self.persona),
            ("human", "{user_prompt}\n\nData: {data}")
        ])
        
        self.chain = self.prompt | self.model | StrOutputParser()
    
    def get_data_from_last_minute(self):
        """Fetches last minute's motor skill data."""
        if not os.path.exists(config.MOTOR_SKILLS_CSV):
            return []
        
        one_minute_ago = datetime.now() - timedelta(minutes=1)
        relevant_data = []

        with open(config.MOTOR_SKILLS_CSV, "r") as f:
            reader = csv.reader(f)
            next(reader)  # Skip header
            for row in reader:
                timestamp = datetime.strptime(row[0], "%Y-%m-%d %H:%M:%S")
                if one_minute_ago <= timestamp <= datetime.now():
                    relevant_data.append({
                        "timestamp": row[0], 
                        "grip_strength": row[1], 
                        "coordination": row[2]
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
            if os.path.exists(config.ANALYZED_MOTOR_SKILLS_JSON):
                with open(config.ANALYZED_MOTOR_SKILLS_JSON, "r") as f:
                    try:
                        previous_results = json.load(f)
                    except json.JSONDecodeError:
                        pass

            previous_results.append(result_data)

            with open(config.ANALYZED_MOTOR_SKILLS_JSON, "w") as f:
                json.dump(previous_results, f, indent=4)

            timestamp = result_data['timestamp']
            print(f"[Motor Skill Agent] Analysis saved at {timestamp}.")
    
    def analyze(self):
        """Runs the motor skill analysis."""
        # Set running flag to prevent multiple concurrent analyses
        if self._running:
            print("[Motor Skill Agent] Analysis already in progress, skipping.")
            return
            
        self._running = True
        
        try:
            # Get latest data
            data = self.get_data_from_last_minute()
            if not data:
                print("[Motor Skill Agent] No data available for analysis.")
                self._running = False
                return
            
            # Create a detailed prompt for analysis
            analysis_prompt = """Please analyze these motor skill measurements.
            Examine both grip strength and coordination values for signs of motor impairment.
            Look for patterns that might indicate Parkinson's disease symptoms such as tremors, rigidity, or bradykinesia.
            Consider whether any changes are asymmetrical (affecting one side more than the other).
            Assess if there are any "on/off" phenomena visible in the data.
            Evaluate the progression or stability of symptoms over time.
            Provide a detailed neurological assessment based on these measurements."""
            
            # Run the analysis through the LLM chain
            analysis = self.chain.invoke({
                "user_prompt": analysis_prompt,
                "data": json.dumps(data, indent=2)
            })
            
            # Save results
            self.save_analysis_results(analysis)
            
            # Update last analysis time
            self._last_analysis_time = datetime.now()
            print(f"[Motor Skill Agent] Completed analysis.")
            
        except Exception as e:
            print(f"[Motor Skill Agent] Error during analysis: {str(e)}")
        finally:
            # Always reset running flag when done
            self._running = False
    
    def run(self):
        """Runs the analysis loop with proper timing."""
        print("[Motor Skill Agent] Started monitoring. Will analyze data every minute.")
        
        while True:
            current_time = datetime.now()
            
            # If it's the first run or at least 1 minute has passed since last analysis
            if (self._last_analysis_time is None or 
                (current_time - self._last_analysis_time).total_seconds() >= 30):
                
                # Start analysis in a separate thread to not block the main loop
                threading.Thread(target=self.analyze, daemon=True).start()
                
            # Sleep for a short time before checking again
            time.sleep(10)  # Check every 10 seconds instead of blocking for a full minute