import threading
from core.agent_manager import start_all_agents
from sensors.blood_pressure_sensor import BloodPressureSensor
from sensors.heart_rate_sensor import HeartRateSensor
# from sensors.motor_skill_sensor import motor_skill_sensor
import time

def main():
    print("Starting multi-agent system...")

    # Start sensor threads FIRST
    print("Starting sensors...")
    bp_sensor = BloodPressureSensor(frequency=10)
    hr_sensor = HeartRateSensor(frequency=10)
    
    bp_sensor.start()
    hr_sensor.start()
    
    time.sleep(5)

    # Now start all agents
    print("Starting agents...")
    start_all_agents()

    # Keep server alive
    while True:
        try:
            # input("Press Enter to continue monitoring (Ctrl+C to stop)...")
            pass
        except KeyboardInterrupt:
            print("Shutting down server...")
            break

if __name__ == "__main__":
    main()
