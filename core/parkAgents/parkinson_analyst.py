import json
import threading
import os
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
        self.file_lock = threading.Lock()  # Lock for thread-safe file operations
        self._last_analysis_time = None    # Track when we last did an analysis
        self._running = False              # Flag to prevent multiple analyses at once
        
        # Define the Parkinson's disease expert persona with JSON output format instructions
        # IMPORTANT: Using double curly braces to escape them for LangChain template
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

                            ALERTS SHOULD BE RAISED WHEN:
                            - Multiple cardinal symptoms of Parkinson's disease are present
                            - Strong correlation between autonomic and motor symptoms
                            - Clear evidence of symptom asymmetry
                            - Progression of symptoms over time
                            - Distinct "on/off" phenomena in motor function
                            - Signs that warrant immediate medical attention or evaluation

                            Always provide a nuanced analysis that considers alternative explanations while highlighting concerning patterns that warrant further investigation.
                            
                            YOU MUST RESPOND IN THE FOLLOWING JSON FORMAT:
                            {{
                                "analysis": "Your detailed analysis text here",
                                "alert": true or false
                            }}
                            
                            Use true for alert if any alert conditions are met, otherwise use false.
                            Do not include any text outside of this JSON structure."""
        
        # Set up the chain with persona
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", self.persona),
            ("human", "{user_prompt}\n\n{data}")
        ])
        
        # Use string output parser as we'll parse the JSON manually
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
    
    def parse_json_response(self, response_text):
        """Parse the JSON response from the LLM. Handle potential formatting issues."""
        try:
            # Find anything that looks like a JSON object in the response
            import re
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                result = json.loads(json_str)
                
                # Validate the result has the required fields
                if 'analysis' in result and 'alert' in result:
                    # Ensure alert is boolean
                    if isinstance(result['alert'], str):
                        result['alert'] = result['alert'].lower() == 'true'
                    return result
            
            # If we reach here, either no JSON was found or it was invalid
            raise ValueError("Response doesn't contain valid JSON with required fields")
            
        except Exception as e:
            print(f"[Parkinson Analyst] Error parsing JSON response: {str(e)}")
            # Create a default result with the original text as analysis and default to alert=True for safety
            return {
                'analysis': response_text,
                'alert': True  # Default to alert=True on parsing error for safety
            }
    
    def save_analysis_results(self, result_data, bp_data, hr_data, ms_data):
        """Saves analysis results to JSON file with thread safety."""
        with self.file_lock:  # Ensure only one thread writes at a time
            output_data = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "blood_pressure": bp_data['analysis'],
                "heart_rate": hr_data['analysis'],
                "motor_skills": ms_data['analysis'] if isinstance(ms_data, dict) else ms_data,
                "parkinson_analysis": result_data.get('analysis', 'No analysis available'),
                "alert": result_data.get('alert', True)  # Default to True if missing
            }

            previous_results = []
            if os.path.exists(config.ANALYZED_PARKINSON_JSON):
                with open(config.ANALYZED_PARKINSON_JSON, "r") as f:
                    try:
                        previous_results = json.load(f)
                    except json.JSONDecodeError:
                        pass  # Handle empty or corrupt file case

            previous_results.append(output_data)

            with open(config.ANALYZED_PARKINSON_JSON, "w") as f:
                json.dump(previous_results, f, indent=4)

            timestamp = output_data['timestamp']
            print(f"[Parkinson Analyst] Analysis saved at {timestamp}.")
    
    def analyze(self):
        """Performs integrated analysis using the latest data from all three sources."""
        # Set running flag to prevent multiple concurrent analyses
        if self._running:
            print("[Parkinson Analyst] Analysis already in progress, skipping.")
            return
            
        self._running = True
        
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
            Provide a detailed assessment with appropriate medical confidence levels.
            
            Remember to return your analysis in the required JSON format with both the analysis text and alert boolean.
            """
            
            # Format the data for the LLM
            formatted_data = f"""Blood Pressure Analysis: {bp_data['analysis']}

                                Heart Rate Analysis: {hr_data['analysis']}

                                Motor Skills Analysis: {ms_data['analysis']}"""
            
            # Run the analysis through the LLM chain
            response = self.chain.invoke({
                "user_prompt": analysis_prompt,
                "data": formatted_data
            })
            
            # Parse the JSON response
            result_data = self.parse_json_response(response)
            
            # Save the analysis results
            self.save_analysis_results(result_data, bp_data, hr_data, ms_data)
            
            # Update last analysis time
            self._last_analysis_time = datetime.now()
            print(f"[Parkinson Analyst] Completed integrated analysis.")
            
        except Exception as e:
            print(f"[Parkinson Analyst] Error during analysis: {str(e)}")
            # Create default result with error message
            if 'bp_data' in locals() and 'hr_data' in locals() and 'ms_data' in locals():
                error_result = {
                    'analysis': f"Error during analysis: {str(e)}",
                    'alert': True  # Default to alert=True on error for safety
                }
                self.save_analysis_results(error_result, bp_data, hr_data, ms_data)
        finally:
            # Always reset running flag when done
            self._running = False
    
    def run(self):
        """Runs the analysis loop with proper timing after initial data is available."""
        # Wait for initial data to be available
        while not self.check_all_data_available():
            print("[Parkinson Analyst] Waiting for data... Checking again in 10 seconds.")
            time.sleep(10)  # Retry in 10 seconds if data is missing

        print("[Parkinson Analyst] Data available. Starting analysis every minute.")
        
        while True:
            current_time = datetime.now()
            
            # If it's the first run or at least 30 seconds have passed since last analysis
            if (self._last_analysis_time is None or 
                (current_time - self._last_analysis_time).total_seconds() >= 30):
                
                # Start analysis in a separate thread to not block the main loop
                threading.Thread(target=self.analyze, daemon=True).start()
                
            # Sleep for a short time before checking again
            time.sleep(10)  # Check every 10 seconds instead of blocking for a full minute