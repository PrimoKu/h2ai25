import threading
from agents.mood_analyst import run_mood_agent
from agents.blood_pressure_analyst import run_bp_agent
from agents.heart_rate_analyst import run_hr_agent
from agents.motor_skill_analyst import run_ms_agent
from agents.parkinson_analyst import run_parkinson_agent

def start_all_agents():
    """Starts all agent threads"""
    threads = [
        threading.Thread(target=run_mood_agent, daemon=True),
        threading.Thread(target=run_bp_agent, daemon=True),
        threading.Thread(target=run_hr_agent, daemon=True),
        threading.Thread(target=run_ms_agent, daemon=True),
        threading.Thread(target=run_parkinson_agent, daemon=True),
    ]
    
    for thread in threads:
        thread.start()

start_all_agents()
