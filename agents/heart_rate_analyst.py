import json
import csv
import threading
import os
import schedule
import time
from datetime import datetime, timedelta
from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent, AgentType
from langchain.tools import Tool
import config

# Initialize OpenAI model
llm = ChatOpenAI(model="gpt-4", temperature=0, openai_api_key=config.OPENAI_API_KEY)

def get_data_from_last_minute():
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
                relevant_data.append({"timestamp": row[0], "heart_rate": row[1]})

    return relevant_data

def analyze_heart_rate():
    """Analyzes last minute's heart rate data."""
    data = get_data_from_last_minute()
    if not data:
        print("[Heart Rate Agent] No data available for analysis.")
        return

    prompt = f"Analyze the following heart rate readings: {data}"
    analysis = llm.predict(prompt)

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

    print(f"[Heart Rate Agent] Analysis saved.")

# Define the tool for the agent
hr_analysis_tool = Tool(
    name="Heart Rate Analysis",
    func=analyze_heart_rate,
    description="You are a heart health expert. Your role is to examine the user's heart rate readings from the last minute, detect irregularities, assess overall heart health, and offer recommendations if needed."
)

# Initialize the agent
hr_agent = initialize_agent(
    tools=[hr_analysis_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

def run_hr_agent():
    """Schedules analysis every minute."""
    schedule.every(1).minutes.do(lambda: threading.Thread(target=hr_agent.run, args=("Analyze the latest data.",)).start())

    while True:
        schedule.run_pending()
        time.sleep(1)
