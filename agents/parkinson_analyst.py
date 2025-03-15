import json
import threading
import os
import schedule
import time
from datetime import datetime
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
import config

class ParkinsonAnalystAgent:
    def __init__(self, model="gpt-4", temperature=0, api_key=None):
        self.api_key = api_key or config.OPENAI_API_KEY
        self.model = ChatOpenAI(model=model, temperature=temperature, api_key=self.api_key)
        
        # Define the Parkinson's disease expert persona
        self.persona = """You are a specialized neurologist with expertise in Parkinson's disease diagnosis and treatment.

                            Your responsibilities:
                            1. Comprehensively analyze multiple data points (blood pressure, heart rate, motor skills) to identify potential Parkinson's disease symptoms
                            2. Correlate autonomic nervous system indicators (blood pressure, heart rate) with motor symptoms
                            3. Distinguish between Parkinson's disease symptoms and other conditions with similar presentations
                            4. Identify early warning signs of Parkinson's disease that might not be obvious when analyzing individual metrics separately
                            5. Provide evidence-based assessment with appropriate confidence levels

                            When analyzing for Parkinson's disease:
                            - Classic motor symptoms include tremor, bradykinesia (slowness of movement), rigidity, and postural instability
                            - Autonomic dysfunction can manifest as orthostatic hypotension or cardiac autonomic dysfunction
                            - Non-motor symptoms include sleep disorders, cognitive changes, and autonomic dysfunction
                            - Asymmetry of symptoms is a hallmark feature (symptoms typically begin on one side)
                            - Symptoms often fluctuate throughout the day and can be affected by medication timing
                            - Combination of motor symptoms with autonomic dysfunction strengthens diagnostic suspicion

                            Always provide a nuanced analysis that considers alternative explanations while highlighting concerning patterns that warrant further investigation."""
        
        # Set up the chain with persona
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", self.persona),
            ("human", "{user_prompt}\n\n{data}")
        ])
        
        self.chain = self.prompt | self.model | StrOutputParser()
    
    def read_latest_entry(self, file_path):
        """Reads the latest entry from a JSON file."""
        if not os.path.exists(file_path):
            return None

        with open(file_path, "r") as f:
            try:
                data = json.load(f)
                if isinstance(data, list) and data:  # Ensure data is a non-empty list
                    return data[-1]  # Return the most recent entry
                else:
                    return None
            except json.JSONDecodeError:
                print(f"[Parkinson Analyst] Error reading {file_path}.")
                return None
    
    def check_all_data_available(self):
        """Checks if all three analyzed JSON files contain at least one valid entry."""
        bp_data = self.read_latest_entry(config.ANALYZED_BLOOD_PRESSURE_JSON)
        hr_data = self.read_latest_entry(config.ANALYZED_HEART_RATE_JSON)
        ms_data = self.read_latest_entry(config.ANALYZED_MOTOR_SKILLS_JSON)

        return all([bp_data, hr_data, ms_data])  # Returns True only if all are available
    
    def save_analysis_results(self, result_data):
        """Saves analysis results to JSON file."""
        previous_results = []
        
        if os.path.exists(config.ANALYZED_PARKINSON_JSON):
            with open(config.ANALYZED_PARKINSON_JSON, "r") as f:
                try:
                    previous_results = json.load(f)
                except json.JSONDecodeError:
                    pass  # Handle empty or corrupt file case

        previous_results.append(result_data)

        with open(config.ANALYZED_PARKINSON_JSON, "w") as f:
            json.dump(previous_results, f, indent=4)

        print(f"[Parkinson Analyst] Analysis saved.")
    
    def analyze(self):
        """Performs integrated analysis using the latest data from all three sources."""
        try:
            if not self.check_all_data_available():
                print("[Parkinson Analyst] Waiting for enough data to be available...")
                return  # Wait until all JSON files have data

            # Retrieve latest data from each file
            bp_data = self.read_latest_entry(config.ANALYZED_BLOOD_PRESSURE_JSON)
            hr_data = self.read_latest_entry(config.ANALYZED_HEART_RATE_JSON)
            ms_data = self.read_latest_entry(config.ANALYZED_MOTOR_SKILLS_JSON)

            # Ensure valid data is retrieved
            if not all([bp_data, hr_data, ms_data]):
                print("[Parkinson Analyst] Skipping analysis due to missing data.")
                return
            
            # Create a comprehensive analysis prompt
            analysis_prompt = """Please conduct an integrated analysis for potential signs of Parkinson's disease using the following three data analyses.
            
            Consider the relationships between blood pressure regulation, heart rate patterns, and motor function.
            Look for correlations that might indicate autonomic dysfunction alongside motor symptoms.
            Assess whether asymmetrical symptoms are present in the motor data.
            Evaluate the combined clinical picture for early or established signs of Parkinson's disease.
            Provide a detailed assessment with appropriate medical confidence levels."""
            
            # Format the data for the LLM
            formatted_data = f"""Blood Pressure Analysis: {bp_data['analysis']}

                                Heart Rate Analysis: {hr_data['analysis']}

                                Motor Skills Analysis: {ms_data['analysis']}"""
            
            # Run the analysis through the LLM chain
            parkinson_analysis = self.chain.invoke({
                "user_prompt": analysis_prompt,
                "data": formatted_data
            })
            
            # Prepare result data
            result_data = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "blood_pressure": bp_data['analysis'],
                "heart_rate": hr_data['analysis'],
                "motor_skills": ms_data['analysis'],
                "parkinson_analysis": parkinson_analysis
            }
            
            # Save the analysis results
            self.save_analysis_results(result_data)
            
            print(f"[Parkinson Analyst] Completed integrated analysis.")
            
        except Exception as e:
            print(f"[Parkinson Analyst] Error during analysis: {str(e)}")
    
    def run(self):
        """Schedules Parkinson analysis every minute after data is available."""
        # Wait for initial data to be available
        while not self.check_all_data_available():
            print("[Parkinson Analyst] Waiting for data... Checking again in 10 seconds.")
            time.sleep(10)  # Retry in 10 seconds if data is missing

        print("[Parkinson Analyst] Data available. Starting analysis every minute.")
        schedule.every(1).minutes.do(lambda: threading.Thread(target=self.analyze).start())

        while True:
            schedule.run_pending()
            time.sleep(1)