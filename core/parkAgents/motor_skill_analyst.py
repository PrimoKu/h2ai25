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

        # Define the motor skill expert persona with JSON output format instructions
        # IMPORTANT: Using double curly braces to escape them for LangChain template
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

                            ALERTS SHOULD BE RAISED WHEN:
                            - Significant decline in grip strength compared to baseline
                            - Clear asymmetry in motor function
                            - Notable tremor patterns visible in coordination measurements
                            - Marked "on/off" fluctuations in motor function
                            - Progressive decline across multiple readings
                            - Any patterns strongly suggestive of Parkinson's disease

                            Always consider both the absolute values and the trends over time, as progressive decline is more concerning than stable readings, even if below normal thresholds.
                            
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
            ("human", "{user_prompt}\n\nData: {data}")
        ])
        
        # Use string output parser as we'll parse the JSON manually
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
    
    def get_data_from_files(self):
        """Fetches the last 100 entries from each motor skill data file."""
        combined_data = {
            "gait": [],
            "postural": [],
            "tremor": []
        }
        
        # Read gait data
        if os.path.exists(config.GAIT_CSV):
            try:
                with open(config.GAIT_CSV, "r") as f:
                    reader = csv.reader(f)
                    header = next(reader)  # Skip header
                    all_rows = list(reader)  # Read all rows
                    # Get the last 100 entries (or all if fewer than 100)
                    last_entries = all_rows[-10:] if len(all_rows) > 100 else all_rows
                    
                    for row in last_entries:
                        if len(row) >= 5:  # Ensure row has time + 4 coordinates
                            try:
                                combined_data["gait"].append({
                                    "time": float(row[0]) if row[0] else None,
                                    "left_foot_x": float(row[1]) if row[1] else None,
                                    "left_foot_y": float(row[2]) if row[2] else None,
                                    "right_foot_x": float(row[3]) if row[3] else None,
                                    "right_foot_y": float(row[4]) if row[4] else None
                                })
                            except (ValueError, IndexError) as e:
                                print(f"[Motor Skill Agent] Error parsing gait data row: {e}")
            except Exception as e:
                print(f"[Motor Skill Agent] Error reading gait data: {e}")
        
        # Read postural data
        if os.path.exists(config.POSTURAL_CSV):
            try:
                with open(config.POSTURAL_CSV, "r") as f:
                    reader = csv.reader(f)
                    header = next(reader)  # Skip header
                    all_rows = list(reader)  # Read all rows
                    # Get the last 100 entries (or all if fewer than 100)
                    last_entries = all_rows[-10:] if len(all_rows) > 100 else all_rows
                    
                    for row in last_entries:
                        if len(row) >= 5:  # Ensure row has time + 4 values
                            try:
                                combined_data["postural"].append({
                                    "time": float(row[0]) if row[0] else None,
                                    "x": int(row[1]) if row[1] else None,
                                    "y": int(row[2]) if row[2] else None,
                                    "sway_distance": float(row[3]) if row[3] else None,
                                    "sway_velocity": float(row[4]) if row[4] else None
                                })
                            except (ValueError, IndexError) as e:
                                print(f"[Motor Skill Agent] Error parsing postural data row: {e}")
            except Exception as e:
                print(f"[Motor Skill Agent] Error reading postural data: {e}")
        
        # Read tremor data
        if os.path.exists(config.TREMOR_CSV):
            try:
                with open(config.TREMOR_CSV, "r") as f:
                    reader = csv.reader(f)
                    header = next(reader)  # Skip header
                    all_rows = list(reader)  # Read all rows
                    # Get the last 100 entries (or all if fewer than 100)
                    last_entries = all_rows[-10:] if len(all_rows) > 100 else all_rows
                    
                    for row in last_entries:
                        if len(row) >= 4:  # Ensure row has time + 3 values
                            try:
                                combined_data["tremor"].append({
                                    "time": float(row[0]) if row[0] else None,
                                    "hand": row[1],
                                    "tremor_frequency": float(row[2]) if row[2] else None,
                                    "tremor_amplitude": float(row[3]) if row[3] else None
                                })
                            except (ValueError, IndexError) as e:
                                print(f"[Motor Skill Agent] Error parsing tremor data row: {e}")
            except Exception as e:
                print(f"[Motor Skill Agent] Error reading tremor data: {e}")
        
        # Check if we have data from any source
        if not any(combined_data.values()):
            print("[Motor Skill Agent] No motor skill data available from any source.")
            return []
        
        # Return a structured object containing all the data
        return {
            "data": combined_data,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
    
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
            print(f"[Motor Skill Agent] Error parsing JSON response: {str(e)}")
            # Create a default result with the original text as analysis and default to alert=True for safety
            return {
                'analysis': response_text,
                'alert': True  # Default to alert=True on parsing error for safety
            }
    
    def save_analysis_results(self, result_data):
        """Saves analysis results to JSON file with thread safety."""
        with self.file_lock:  # Ensure only one thread writes at a time
            output_data = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "analysis": result_data.get('analysis', 'No analysis available'),
                "alert": result_data.get('alert', True)  # Default to True if missing
            }

            previous_results = []
            if os.path.exists(config.ANALYZED_MOTOR_SKILLS_JSON):
                with open(config.ANALYZED_MOTOR_SKILLS_JSON, "r") as f:
                    try:
                        previous_results = json.load(f)
                    except json.JSONDecodeError:
                        pass

            previous_results.append(output_data)

            with open(config.ANALYZED_MOTOR_SKILLS_JSON, "w") as f:
                json.dump(previous_results, f, indent=4)

            timestamp = output_data['timestamp']
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
            data = self.get_data_from_files()
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
            Provide a detailed neurological assessment based on these measurements.
            
            Remember to return your analysis in the required JSON format with both the analysis text and alert boolean.
            """
            
            # Run the analysis through the LLM chain
            response = self.chain.invoke({
                "user_prompt": analysis_prompt,
                "data": json.dumps(data, indent=2)
            })
            
            # Parse the JSON response
            result_data = self.parse_json_response(response)
            
            # Save results
            self.save_analysis_results(result_data)
            
            # Update last analysis time
            self._last_analysis_time = datetime.now()
            print(f"[Motor Skill Agent] Completed analysis.")
            
        except Exception as e:
            print(f"[Motor Skill Agent] Error during analysis: {str(e)}")
            # Create default result with error message
            error_result = {
                'analysis': f"Error during analysis: {str(e)}",
                'alert': True  # Default to alert=True on error for safety
            }
            self.save_analysis_results(error_result)
        finally:
            # Always reset running flag when done
            self._running = False
    
    def run(self):
        """Runs the analysis loop with proper timing."""
        print("[Motor Skill Agent] Started monitoring. Will analyze data every minute.")
        
        while True:
            current_time = datetime.now()
            
            # If it's the first run or at least 30 seconds have passed since last analysis
            if (self._last_analysis_time is None or 
                (current_time - self._last_analysis_time).total_seconds() >= 30):
                
                # Start analysis in a separate thread to not block the main loop
                threading.Thread(target=self.analyze, daemon=True).start()
                
            # Sleep for a short time before checking again
            time.sleep(10)  # Check every 10 seconds instead of blocking for a full minute