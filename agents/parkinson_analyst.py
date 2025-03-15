import json
import threading
import os
import schedule
import time
from datetime import datetime
from langchain.chat_models import ChatOpenAI
import config

llm = ChatOpenAI(model="gpt-4", temperature=0, openai_api_key=config.OPENAI_API_KEY)

def read_latest_entry(file_path):
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
            print(f"[Parkinson Agent] Error reading {file_path}.")
            return None

def check_all_data_available():
    """Checks if all three analyzed JSON files contain at least one valid entry."""
    bp_data = read_latest_entry(config.ANALYZED_BLOOD_PRESSURE_JSON)
    hr_data = read_latest_entry(config.ANALYZED_HEART_RATE_JSON)
    ms_data = read_latest_entry(config.ANALYZED_MOTOR_SKILLS_JSON)

    return all([bp_data, hr_data, ms_data])  # Returns True only if all are available

def analyze_parkinson():
    """Performs analysis using the latest data from all three JSON files."""
    if not check_all_data_available():
        print("[Parkinson Agent] Waiting for enough data to be available...")
        return  # Wait until all JSON files have data

    # Retrieve latest data from each file
    bp_data = read_latest_entry(config.ANALYZED_BLOOD_PRESSURE_JSON)
    hr_data = read_latest_entry(config.ANALYZED_HEART_RATE_JSON)
    ms_data = read_latest_entry(config.ANALYZED_MOTOR_SKILLS_JSON)

    # Ensure valid data is retrieved
    if not all([bp_data, hr_data, ms_data]):
        print("[Parkinson Agent] Skipping analysis due to missing data.")
        return

    # Construct analysis prompt
    prompt = (
        f"Analyze the following patient data for Parkinson's disease symptoms:\n"
        f"Blood Pressure Analysis: {bp_data['analysis']}\n"
        f"Heart Rate Analysis: {hr_data['analysis']}\n"
        f"Motor Skills Analysis: {ms_data['analysis']}"
    )

    # Perform AI-based analysis
    analysis = llm.predict(prompt)

    # Store results with timestamps
    result_data = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "blood_pressure": bp_data['analysis'],
        "heart_rate": hr_data['analysis'],
        "motor_skills": ms_data['analysis'],
        "parkinson_analysis": analysis
    }

    # Save analysis to JSON file
    result_file = config.DATA_STORAGE_PATH + "parkinson_analysis.json"
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

    print(f"[Parkinson Agent] Analysis saved.")

def run_parkinson_agent():
    """Schedules Parkinson analysis every minute without overlapping."""
    while not check_all_data_available():
        print("[Parkinson Agent] Waiting for data... Checking again in 10 seconds.")
        time.sleep(10)  # Retry in 10 seconds if data is missing

    print("[Parkinson Agent] Data available. Starting analysis every minute.")
    schedule.every(1).minutes.do(analyze_parkinson)

    while True:
        schedule.run_pending()
        time.sleep(1)

parkinson_agent_thread = threading.Thread(target=run_parkinson_agent, daemon=True)
parkinson_agent_thread.start()
