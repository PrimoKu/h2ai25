import json
import threading
import os
import schedule
import time
from datetime import datetime
from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent, AgentType
from langchain.tools import Tool
import config

# Initialize OpenAI model
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
        f"You are a Parkinson’s disease analyst. Your task is to analyze the following patient data and "
        f"identify potential symptoms or warning signs of Parkinson’s disease:\n\n"
        f"Blood Pressure Analysis: {bp_data['analysis']}\n"
        f"Heart Rate Analysis: {hr_data['analysis']}\n"
        f"Motor Skills Analysis: {ms_data['analysis']}\n\n"
        f"Based on these readings, provide a detailed assessment of whether there are any Parkinson’s indicators."
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

# Define the tool for the agent
parkinson_analysis_tool = Tool(
    name="Parkinson's Disease Analysis",
    func=analyze_parkinson,
    description="You are a Parkinson’s disease analyst. Your job is to evaluate recent blood pressure, heart rate, and motor skill data to detect possible symptoms of Parkinson’s disease. Analyze the data and provide an expert assessment."
)

# Initialize the agent
parkinson_agent = initialize_agent(
    tools=[parkinson_analysis_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

def run_parkinson_agent():
    """Schedules Parkinson analysis every minute."""
    while not check_all_data_available():
        print("[Parkinson Agent] Waiting for data... Checking again in 10 seconds.")
        time.sleep(10)  # Retry in 10 seconds if data is missing

    print("[Parkinson Agent] Data available. Starting analysis every minute.")
    schedule.every(1).minutes.do(lambda: threading.Thread(target=parkinson_agent.run, args=("Analyze the latest data.",)).start())

    while True:
        schedule.run_pending()
        time.sleep(1)

parkinson_agent_thread = threading.Thread(target=run_parkinson_agent, daemon=True)
parkinson_agent_thread.start()
