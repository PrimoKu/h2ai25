import threading
from core.agent_manager import start_all_agents
from sensors.blood_pressure_sensor import blood_pressure_sensor
from sensors.heart_rate_sensor import heart_rate_sensor
from sensors.motor_skill_sensor import motor_skill_sensor

def main():
    print("Starting multi-agent system...")

    # Start all agents
    start_all_agents()

    # Start sensor threads
    threading.Thread(target=blood_pressure_sensor, daemon=True).start()
    threading.Thread(target=heart_rate_sensor, daemon=True).start()
    threading.Thread(target=motor_skill_sensor, daemon=True).start()

    while True:
        try:
            input("Press Enter to continue monitoring (Ctrl+C to stop)...")
        except KeyboardInterrupt:
            print("Shutting down server...")
            break

if __name__ == "__main__":
    main()