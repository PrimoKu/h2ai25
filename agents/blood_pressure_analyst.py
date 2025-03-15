import json
import csv
import threading
import os
import schedule
import time
from datetime import datetime, timedelta
from langchain.chat_models import ChatOpenAI
import config

# Initialize OpenAI model
llm = ChatOpenAI(model="gpt-4", temperature=0, openai_api_key=config.OPENAI_API_KEY)

def get_data_from_last_minute():
    """Fetches blood pressure data from one minute ago."""
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
                relevant_data.append({"timestamp": row[0], "systolic": row[1], "diastolic": row[2]})

    return relevant_data

def analyze_blood_pressure():
    """Analyzes last minute's blood pressure data."""
    data = get_data_from_last_minute()
    if not data:
        print("[Blood Pressure Agent] No data available for analysis.")
        return

    prompt = f"Analyze the following blood pressure readings: {data}"
    analysis = llm.predict(prompt)

    # Store results with timestamps
    result_data = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "analysis": analysis
    }

    # Load previous results and append new entry
    previous_results = []
    if os.path.exists(config.ANALYZED_BLOOD_PRESSURE_JSON):
        with open(config.ANALYZED_BLOOD_PRESSURE_JSON, "r") as f:
            try:
                previous_results = json.load(f)
            except json.JSONDecodeError:
                pass  # Handle empty or corrupt file case

    previous_results.append(result_data)

    with open(config.ANALYZED_BLOOD_PRESSURE_JSON, "w") as f:
        json.dump(previous_results, f, indent=4)

    print(f"[Blood Pressure Agent] Analysis saved.")

def run_bp_agent():
    """Schedules analysis every minute without skipping."""
    schedule.every(1).minutes.do(lambda: threading.Thread(target=analyze_blood_pressure).start())

    while True:
        schedule.run_pending()
        time.sleep(1)

bp_agent_thread = threading.Thread(target=run_bp_agent, daemon=True)
bp_agent_thread.start()
