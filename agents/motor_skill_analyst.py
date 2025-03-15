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

# Define the tool for the agent
motor_skill_analysis_tool = Tool(
    name="Motor Skill Analysis",
    func=analyze_motor_skills,
    description="You are a neurological and motor function analyst. Your job is to evaluate the user's grip strength and coordination based on last-minute data. Detect any signs of motor impairment, fatigue, or potential neurological disorders."
)


# Initialize the agent
motor_skill_agent = initialize_agent(
    tools=[motor_skill_analysis_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

def run_ms_agent():
    """Schedules analysis every minute."""
    schedule.every(1).minutes.do(lambda: threading.Thread(target=motor_skill_agent.run, args=("Analyze the latest data.",)).start())

    while True:
        schedule.run_pending()
        time.sleep(1)
