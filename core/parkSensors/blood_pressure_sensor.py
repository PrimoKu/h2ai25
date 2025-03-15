import csv
import time
import random
import os
import threading
from datetime import datetime
import config

class BloodPressureSensor:
    """
    Simulates blood pressure readings and writes them to a CSV file.
    Generates realistic blood pressure values with small variations
    and occasional anomalies to test the analysis agent.
    """
    
    def __init__(self, frequency=10):
        """
        Initialize the blood pressure sensor simulator.
        
        Args:
            frequency (int): How often to generate readings, in seconds
        """
        self.frequency = frequency
        self.running = False
        self.thread = None
        
        # Ensure the data directory exists
        os.makedirs(os.path.dirname(config.BLOOD_PRESSURE_CSV), exist_ok=True)
        
        # Check if file exists and create with header if it doesn't
        file_exists = os.path.isfile(config.BLOOD_PRESSURE_CSV)
        
        if not file_exists:
            with open(config.BLOOD_PRESSURE_CSV, 'w', newline='') as csvfile:
                fieldnames = ['timestamp', 'systolic', 'diastolic']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                print(f"[Blood Pressure Sensor] Created new file: {config.BLOOD_PRESSURE_CSV}")
    
    def start(self):
        """Start the blood pressure sensor simulation in a separate thread."""
        if self.running:
            print("[Blood Pressure Sensor] Already running.")
            return
            
        self.running = True
        self.thread = threading.Thread(target=self._run_simulation)
        self.thread.daemon = True
        self.thread.start()
        print(f"[Blood Pressure Sensor] Started simulation (frequency: {self.frequency}s)")
        
        return self.thread
    
    def stop(self):
        """Stop the blood pressure sensor simulation."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=self.frequency + 1)
            print("[Blood Pressure Sensor] Stopped simulation.")
    
    def _run_simulation(self):
        """Run the actual simulation loop."""
        # Simulation parameters
        base_systolic = 120  # Normal systolic baseline
        base_diastolic = 80  # Normal diastolic baseline
        
        # Simulation variations
        variations = [
            # (hours active, systolic change, diastolic change, description)
            (6, 0, 0, "normal baseline"),
            (3, 15, 5, "mild elevation"),
            (1, -10, -5, "below normal"),
            (1, 30, 15, "high elevation"),
            (0.5, 40, 20, "hypertensive"),
            (12, 5, 5, "slightly elevated")
        ]
        
        # Time tracking for variation changes
        current_variation = 0
        variation_time_left = variations[current_variation][0] * 60 * 60  # Convert hours to seconds
        
        try:
            while self.running:
                # Get current time
                current_time = datetime.now()
                timestamp = current_time.strftime("%Y-%m-%d %H:%M:%S")
                
                # Decrease variation time and change if needed
                variation_time_left -= self.frequency
                if variation_time_left <= 0:
                    current_variation = (current_variation + 1) % len(variations)
                    variation_time_left = variations[current_variation][0] * 60 * 60
                    print(f"[Blood Pressure Sensor] Switching to {variations[current_variation][3]} pattern")
                
                # Calculate BP with current variation plus some noise
                systolic_change = variations[current_variation][1]
                diastolic_change = variations[current_variation][2]
                
                # Add random noise (±5 systolic, ±3 diastolic)
                systolic_noise = random.randint(-5, 5)
                diastolic_noise = random.randint(-3, 3)
                
                # Calculate final values with realistic constraints
                systolic = base_systolic + systolic_change + systolic_noise
                diastolic = base_diastolic + diastolic_change + diastolic_noise
                
                # Ensure diastolic is always less than systolic by at least 10
                if diastolic >= (systolic - 10):
                    diastolic = systolic - 10 - random.randint(0, 10)
                
                # Ensure values are in reasonable ranges
                systolic = max(70, min(200, systolic))
                diastolic = max(40, min(120, diastolic))
                
                # Write data to CSV
                with open(config.BLOOD_PRESSURE_CSV, 'a', newline='') as csvfile:
                    fieldnames = ['timestamp', 'systolic', 'diastolic']
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writerow({
                        'timestamp': timestamp,
                        'systolic': systolic,
                        'diastolic': diastolic
                    })
                
                # print(f"[Blood Pressure Sensor] Recorded: {timestamp}, {systolic}/{diastolic} mmHg")
                
                # Wait before next reading
                time.sleep(self.frequency)
                
        except Exception as e:
            print(f"[Blood Pressure Sensor] Error: {str(e)}")
            self.running = False

# For backwards compatibility and direct script execution
def blood_pressure_sensor(frequency=10):
    """Legacy function to start the blood pressure sensor with default frequency."""
    sensor = BloodPressureSensor(frequency=frequency)
    sensor.start()
    try:
        # Keep the main thread alive
        while sensor.running:
            time.sleep(1)
    except KeyboardInterrupt:
        sensor.stop()