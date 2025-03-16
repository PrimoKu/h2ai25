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
        
        # Define the speech analyst persona with JSON output format instructions
        # IMPORTANT: Using double curly braces to escape them for LangChain template
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

                            ALERTS SHOULD BE RAISED WHEN:
                            - Volume is significantly reduced (hypophonia), which may indicate Parkinson's disease
                            - Jitter or shimmer values are abnormally high (above 0.5), indicating vocal instability
                            - Fundamental frequency shows unusual patterns or monotonicity
                            - Emotional analysis indicates extreme negative emotions (high anger, fear, or sadness)
                            - Multiple speech parameters deviate from normal ranges simultaneously
                            - Clear patterns of vocal degradation are observed over time

                            YOU MUST RESPOND IN THE FOLLOWING JSON FORMAT:
                            {{
                                "analysis": "Your detailed analysis text here",
                                "alert": true or false
                            }}

                            Use true for alert if any alert conditions are met, otherwise use false.
                            Do not include any text outside of this JSON structure.
                            """
        # Set up the prompt chain with the defined persona
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", self.persona),
            ("human", "{user_prompt}\n\nData: {data}")
        ])
        
        # Use string output parser as we'll parse the JSON manually
        self.chain = self.prompt | self.model | StrOutputParser()
    
    def get_data_from_last_minute(self):
        """Fetches last minute's speech data from CSV."""
        if not os.path.exists(config.SPEECH_DATA_CSV):
            print(f"[Speech Analyst Agent] Speech data file not found: {config.SPEECH_DATA_CSV}")
            return []
        
        one_minute_ago = datetime.now() - timedelta(minutes=1)
        relevant_data = []
        
        try:
            with open(config.SPEECH_DATA_CSV, "r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        timestamp = datetime.strptime(row["timestamp"], "%Y-%m-%d %H:%M:%S")
                        if one_minute_ago <= timestamp <= datetime.now():
                            # Process the row with proper value handling
                            processed_row = {}
                            for key, value in row.items():
                                if key == "timestamp" or key == "Volumn Status":
                                    # Keep these as strings
                                    processed_row[key] = value
                                elif key == "Fundamental Frequency(Hz)":
                                    # This is a frequency in Hz, convert to float
                                    try:
                                        processed_row[key] = float(value) if value else None
                                    except ValueError:
                                        processed_row[key] = None
                                        print(f"[Speech Analyst Agent] Error converting Fundamental Frequency: {value}")
                                else:
                                    # All other values are already between 0-1
                                    try:
                                        processed_row[key] = float(value) if value else None
                                    except ValueError:
                                        processed_row[key] = None
                                        print(f"[Speech Analyst Agent] Error converting {key}: {value}")
                            
                            relevant_data.append(processed_row)
                    except (ValueError, KeyError) as e:
                        print(f"[Speech Analyst Agent] Error parsing row: {e}")
                        continue
        except Exception as e:
            print(f"[Speech Analyst Agent] Error reading speech data: {e}")
        
        return relevant_data
    
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
            print(f"[Speech Analyst Agent] Error parsing JSON response: {str(e)}")
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
            if os.path.exists(config.ANALYZED_SPEECH_JSON):
                with open(config.ANALYZED_SPEECH_JSON, "r") as f:
                    try:
                        previous_results = json.load(f)
                    except json.JSONDecodeError:
                        pass

            previous_results.append(output_data)

            with open(config.ANALYZED_SPEECH_JSON, "w") as f:
                json.dump(previous_results, f, indent=4)
    
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
            Provide insights into any anomalies, potential vocal health issues, or emotional cues, and suggest any follow-up actions if necessary.
            
            Remember to return your analysis in the required JSON format with both the analysis text and alert boolean.
            """
            
            # Run the analysis using the LLM chain
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
            print(f"[Speech Analyst Agent] Completed analysis.")
            
        except Exception as e:
            print(f"[Speech Analyst Agent] Error during analysis: {str(e)}")
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