import threading
from core.parkAgents.blood_pressure_analyst import BloodPressureAgent
from core.parkAgents.heart_rate_analyst import HeartRateAgent
from core.parkAgents.motor_skill_analyst import MotorSkillAgent
from core.parkAgents.parkinson_analyst import ParkinsonAnalystAgent
from core.parkAgents.speech_analyst import SpeechAnalystAgent

# Global agent instances to ensure they are only created once
_agent_instances = {
    'bp_agent': None,
    'hr_agent': None,
    'ms_agent': None,
    'pk_agent': None,
    'sp_agent': None
}

def start_all_agents():
    """Starts all agent threads using the singleton approach to prevent duplicate instances"""
    
    # Initialize agents only if they don't exist yet
    if _agent_instances['bp_agent'] is None:
        _agent_instances['bp_agent'] = BloodPressureAgent()
        _agent_instances['hr_agent'] = HeartRateAgent()
        _agent_instances['ms_agent'] = MotorSkillAgent()
        _agent_instances['pk_agent'] = ParkinsonAnalystAgent()
        _agent_instances['sp_agent'] = SpeechAnalystAgent()
    
        # Create threads for each agent's run method
        threads = [
            threading.Thread(target=_agent_instances['bp_agent'].run, daemon=True),
            threading.Thread(target=_agent_instances['hr_agent'].run, daemon=True),
            threading.Thread(target=_agent_instances['ms_agent'].run, daemon=True),
            threading.Thread(target=_agent_instances['pk_agent'].run, daemon=True),
            threading.Thread(target=_agent_instances['sp_agent'].run, daemon=True),
        ]
        
        # Start all threads
        for thread in threads:
            thread.start()
        
        print("All agent threads started successfully.")
        return threads
    else:
        print("Agents already running, no new threads started.")
        return []