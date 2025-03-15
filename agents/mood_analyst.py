import threading
import time
from langchain.chat_models import ChatOpenAI
from langchain.agents import AgentType, initialize_agent
from langchain.tools import Tool
from core.memory import memory_manager

llm = ChatOpenAI(model="gpt-4", temperature=0)

def analyze_mood(patient_id, data):
    memory = memory_manager.load_memory(patient_id)
    memory.save_context({"input": data}, {"output": "Mood analysis performed."})
    memory_manager.save_memory(patient_id, memory)
    return f"Analyzing mood for patient {patient_id}: {data}"

mood_tool = Tool(
    name="MoodAnalyzer",
    func=lambda data: analyze_mood("patient_1", data),  # Replace dynamically
    description="Analyzes mood based on given patient data."
)

mood_agent = initialize_agent(
    tools=[mood_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

def run_mood_agent():
    """Continuously runs the mood analyst agent"""
    while True:
        if not data_queue.is_empty():
            patient_id, data = data_queue.get_data()
            result = mood_agent.run(data)
            print(f"[Mood Agent] {result}")
        time.sleep(1)  # Prevent excessive CPU usage

mood_agent_thread = threading.Thread(target=run_mood_agent, daemon=True)
mood_agent_thread.start()
