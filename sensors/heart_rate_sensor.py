import csv
import time
import random
import os
import math
import threading
from datetime import datetime
import config

class HeartRateSensor:
    """
    Simulates heart rate readings and writes them to a CSV file.
    Generates realistic heart rate patterns with variations based on
    time of day, activity levels, and occasional anomalies to test
    the heart rate analysis agent.
    """
    
    def __init__(self, frequency=10):
        """
        Initialize the heart rate sensor simulator.
        
        Args:
            frequency (int): How often to generate readings, in seconds
        """
        self.frequency = frequency
        self.running = False
        self.thread = None
        
        # Ensure the data directory exists
        os.makedirs(os.path.dirname(config.HEART_RATE_CSV), exist_ok=True)
        
        # Check if file exists and create with header if it doesn't
        file_exists = os.path.isfile(config.HEART_RATE_CSV)
        
        if not file_exists:
            with open(config.HEART_RATE_CSV, 'w', newline='') as csvfile:
                fieldnames = ['timestamp', 'heart_rate']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                print(f"[Heart Rate Sensor] Created new file: {config.HEART_RATE_CSV}")
    
    def start(self):
        """Start the heart rate sensor simulation in a separate thread."""
        if self.running:
            print("[Heart Rate Sensor] Already running.")
            return
            
        self.running = True
        self.thread = threading.Thread(target=self._run_simulation)
        self.thread.daemon = True
        self.thread.start()
        print(f"[Heart Rate Sensor] Started simulation (frequency: {self.frequency}s)")
        
        return self.thread
    
    def stop(self):
        """Stop the heart rate sensor simulation."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=self.frequency + 1)
            print("[Heart Rate Sensor] Stopped simulation.")
    
    def _run_simulation(self):
        """Run the actual simulation loop."""
        # Simulation parameters
        base_heart_rate = 72  # Normal resting heart rate baseline
        
        # Heart rate patterns (each pattern runs for a specified duration)
        patterns = [
            # (hours active, base_hr_change, variation_amount, description)
            (4, 0, 5, "resting baseline"),             # Normal resting
            (1, 20, 10, "light activity"),             # Light exercise
            (0.5, 50, 15, "moderate exercise"),        # Moderate exercise
            (0.25, 80, 20, "intense exercise"),        # Intense exercise
            (2, -5, 3, "sleeping"),                    # Sleep
            (0.1, 30, 25, "stress response"),          # Stress
            (0.05, 40, 30, "cardiac anomaly"),         # Brief anomaly
            (3, 5, 8, "daily activities")              # Regular daily activities
        ]
        
        # Time tracking for pattern changes
        current_pattern = 0
        pattern_time_left = patterns[current_pattern][0] * 60 * 60  # Convert hours to seconds
        
        # For natural trends (gradual changes)
        current_hr_offset = 0
        target_hr_offset = 0
        transition_steps = 0
        steps_remaining = 0
        
        try:
            while self.running:
                # Get current time
                current_time = datetime.now()
                timestamp = current_time.strftime("%Y-%m-%d %H:%M:%S")
                
                # Decrease pattern time and change if needed
                pattern_time_left -= self.frequency
                if pattern_time_left <= 0:
                    current_pattern = (current_pattern + 1) % len(patterns)
                    pattern_time_left = patterns[current_pattern][0] * 60 * 60
                    
                    # Set up transition to new pattern
                    target_hr_offset = patterns[current_pattern][1]
                    transition_steps = min(20, int(pattern_time_left / self.frequency / 4))  # Use up to 25% of time for transition
                    steps_remaining = transition_steps
                    
                    print(f"[Heart Rate Sensor] Switching to {patterns[current_pattern][3]} pattern")
                
                # Handle smooth transitions between patterns
                if steps_remaining > 0:
                    current_hr_offset += (target_hr_offset - current_hr_offset) / steps_remaining
                    steps_remaining -= 1
                
                # Calculate heart rate with current pattern
                hr_change = current_hr_offset
                hr_variation = patterns[current_pattern][2]
                
                # Add variation with sinusoidal component for natural rhythm
                time_factor = time.time() / 30  # Slow oscillation
                natural_variation = math.sin(time_factor) * (hr_variation / 2)
                random_variation = random.uniform(-hr_variation/2, hr_variation/2)
                
                # Calculate final heart rate with realistic constraints
                heart_rate = round(base_heart_rate + hr_change + natural_variation + random_variation)
                
                # Ensure values are in reasonable ranges (40-200 bpm)
                heart_rate = max(40, min(200, heart_rate))
                
                # Write data to CSV
                with open(config.HEART_RATE_CSV, 'a', newline='') as csvfile:
                    fieldnames = ['timestamp', 'heart_rate']
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writerow({
                        'timestamp': timestamp,
                        'heart_rate': heart_rate
                    })
                
                # print(f"[Heart Rate Sensor] Recorded: {timestamp}, {heart_rate} BPM")
                
                # Wait before next reading
                time.sleep(self.frequency)
                
        except Exception as e:
            print(f"[Heart Rate Sensor] Error: {str(e)}")
            self.running = False

# For backwards compatibility and direct script execution
def heart_rate_sensor(frequency=10):
    """Legacy function to start the heart rate sensor with default frequency."""
    sensor = HeartRateSensor(frequency=frequency)
    sensor.start()
    try:
        # Keep the main thread alive
        while sensor.running:
            time.sleep(1)
    except KeyboardInterrupt:
        sensor.stop()