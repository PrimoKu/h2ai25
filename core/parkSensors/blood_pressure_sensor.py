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
    
    def __init__(self, frequency=10, include_abnormal=True):
        """
        Initialize the blood pressure sensor simulator.
        
        Args:
            frequency (int): How often to generate readings, in seconds
            include_abnormal (bool): Whether to include abnormal readings that should trigger alerts
        """
        self.frequency = frequency
        self.running = False
        self.thread = None
        self.include_abnormal = include_abnormal
        
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
        
        # Simulation variations: [hours_active, systolic_change, diastolic_change, description, should_trigger_alert]
        normal_variations = [
            (6, 0, 0, "normal baseline", False),
            (3, 15, 5, "mild elevation", False),
            (1, -10, -5, "below normal", False),
            (1, 30, 15, "high elevation", False),
            (0.5, 40, 20, "hypertensive", True),
            (12, 5, 5, "slightly elevated", False)
        ]
        
        # Abnormal variations that should trigger alerts for testing
        abnormal_variations = [
            (0.3, 60, 30, "stage 2 hypertension", True),              # Stage 2 Hypertension
            (0.2, 80, 40, "severe hypertension", True),               # Severe Hypertension
            (0.1, 100, 50, "hypertensive crisis", True),              # Hypertensive Crisis
            (0.05, 140, 60, "extreme hypertensive crisis", True),     # Extreme Hypertensive Crisis
            (0.5, -15, 35, "isolated diastolic hypertension", True),  # Isolated Diastolic Hypertension
            (0.3, 65, -10, "isolated systolic hypertension", True),   # Isolated Systolic Hypertension
            (0.4, 30, 30, "progressive elevation pattern", True)      # Progressive Elevation
        ]
        
        # Combine variations based on configuration
        variations = normal_variations
        if self.include_abnormal:
            variations = normal_variations + abnormal_variations
        
        # Time tracking for variation changes
        current_variation = 0
        variation_time_left = variations[current_variation][0] * 60 * 60  # Convert hours to seconds
        
        # For progressive patterns
        progressive_count = 0
        progressive_increase = 0
        
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
                    
                    # Reset progressive counters when switching variations
                    progressive_count = 0
                    progressive_increase = 0
                    
                    # print(f"[Blood Pressure Sensor] Switching to {variations[current_variation][3]} pattern")
                
                # Calculate BP with current variation plus some noise
                systolic_change = variations[current_variation][1]
                diastolic_change = variations[current_variation][2]
                description = variations[current_variation][3]
                
                # For progressive pattern, gradually increase values
                if description == "progressive elevation pattern":
                    progressive_count += 1
                    if progressive_count % 5 == 0:  # Every 5 readings
                        progressive_increase += 2  # Increase by 2 mmHg
                    
                    systolic_change += progressive_increase
                    diastolic_change += progressive_increase
                
                # Add random noise (±5 systolic, ±3 diastolic)
                systolic_noise = random.randint(-5, 5)
                diastolic_noise = random.randint(-3, 3)
                
                # Calculate final values with realistic constraints
                systolic = base_systolic + systolic_change + systolic_noise
                diastolic = base_diastolic + diastolic_change + diastolic_noise
                
                # Ensure diastolic is always less than systolic by at least 10
                # Exception: for isolated diastolic/systolic patterns, we allow extreme values
                if "isolated" not in description and diastolic >= (systolic - 10):
                    diastolic = systolic - 10 - random.randint(0, 10)
                
                # Ensure values are in reasonable ranges, but allow extreme values for abnormal patterns
                if "crisis" in description:
                    # For crisis patterns, allow more extreme values
                    systolic = max(70, min(250, systolic))
                    diastolic = max(40, min(130, diastolic))
                else:
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
                
                # Wait before next reading
                time.sleep(self.frequency)
                
        except Exception as e:
            print(f"[Blood Pressure Sensor] Error: {str(e)}")
            self.running = False

# For backwards compatibility and direct script execution
def blood_pressure_sensor(frequency=10, include_abnormal=True):
    """Legacy function to start the blood pressure sensor with default frequency."""
    sensor = BloodPressureSensor(frequency=frequency, include_abnormal=include_abnormal)
    sensor.start()
    try:
        # Keep the main thread alive
        while sensor.running:
            time.sleep(1)
    except KeyboardInterrupt:
        sensor.stop()