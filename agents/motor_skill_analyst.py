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
    """Fetches motor skill data from one minute ago."""
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
                relevant_data.append({"timestamp": row[0], "grip_strength": row[1], "coordination": row[2]})

    return relevant_data

def analyze_motor_skills():
    """Analyzes last minute's motor skill data."""
    data = get_data_from_last_minute()
    if not data:
        print("[Motor Skill Agent] No data available for analysis.")
        return

    prompt = f"Analyze the following motor skill readings: {data}"
    analysis = llm.predict(prompt)

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

    print(f"[Motor Skill Agent] Analysis saved.")

def run_ms_agent():
    """Schedules analysis every minute without skipping."""
    schedule.every(1).minutes.do(lambda: threading.Thread(target=analyze_motor_skills).start())

    while True:
        schedule.run_pending()
        time.sleep(1)

ms_agent_thread = threading.Thread(target=run_ms_agent, daemon=True)
ms_agent_thread.start()
