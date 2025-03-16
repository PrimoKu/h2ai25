import threading

from core.parkAgents.agent_manager import start_all_agents
from core.parkSensors.blood_pressure_sensor import BloodPressureSensor
from core.parkSensors.heart_rate_sensor import HeartRateSensor

# from core.parkCam.monitoring.gait import parkCamGait
# from core.parkCam.monitoring.posture import parkCamPosture
# from core.parkCam.monitoring.tremor import parkCamTremor

import subprocess
import time

def run_external_script(path):
    """Runs an external Python script without blocking the main program."""
    subprocess.Popen(["python", path])  # Runs script in a new process

def main():
    print("Starting multi-agent system...")

    # Start sensor threads FIRST
    print("Starting sensors...")
    bp_sensor = BloodPressureSensor(frequency=1)
    hr_sensor = HeartRateSensor(frequency=1)
    
    bp_sensor.start()
    hr_sensor.start()

    print("Starting motor function monitoring...")
    run_external_script("core/parkCam/monitoring/gait.py")
    run_external_script("core/parkCam/monitoring/tremor.py")
    run_external_script("core/parkCam/monitoring/posture.py")

    print("Starting cognitive agent...")
    run_external_script("cognitive_assessment.py")
    
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
