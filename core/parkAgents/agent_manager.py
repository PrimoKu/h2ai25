import threading
from core.parkAgents.blood_pressure_analyst import BloodPressureAgent
from core.parkAgents.heart_rate_analyst import HeartRateAgent
from core.parkAgents.motor_skill_analyst import MotorSkillAgent
from core.parkAgents.parkinson_analyst import ParkinsonAnalystAgent

def start_all_agents():
    """Starts all agent threads using the new class-based approach"""
    
    # Initialize all agents
    bp_agent = BloodPressureAgent()
    hr_agent = HeartRateAgent()
    ms_agent = MotorSkillAgent()
    pk_agent = ParkinsonAnalystAgent()
    
    # Create threads for each agent's run method
    threads = [
        threading.Thread(target=bp_agent.run, daemon=True),
        threading.Thread(target=hr_agent.run, daemon=True),
        threading.Thread(target=ms_agent.run, daemon=True),
        threading.Thread(target=pk_agent.run, daemon=True),
    ]
    
    # Start all threads
    for thread in threads:
        thread.start()
    
    return threads