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
    
    def __init__(self, frequency=10, include_abnormal=True):
        """
        Initialize the heart rate sensor simulator.
        
        Args:
            frequency (int): How often to generate readings, in seconds
            include_abnormal (bool): Whether to include abnormal readings that should trigger alerts
        """
        self.frequency = frequency
        self.running = False
        self.thread = None
        self.include_abnormal = include_abnormal
        
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
        
        # Heart rate patterns: [hours_active, base_hr_change, variation_amount, description, should_trigger_alert]
        normal_patterns = [
            (4, 0, 5, "resting baseline", False),          # Normal resting
            (1, 20, 10, "light activity", False),          # Light exercise
            (0.5, 50, 15, "moderate exercise", False),     # Moderate exercise
            (0.25, 80, 20, "intense exercise", False),     # Intense exercise
            (2, -5, 3, "sleeping", False),                 # Sleep
            (0.1, 30, 25, "stress response", False),       # Stress
            (0.05, 40, 30, "cardiac anomaly", True),       # Brief anomaly
            (3, 5, 8, "daily activities", False)           # Regular daily activities
        ]
        
        # Abnormal patterns that should trigger alerts for testing
        abnormal_patterns = [
            (0.2, 55, 5, "sustained tachycardia", True),       # Sustained tachycardia (>120 BPM)
            (0.15, -30, 3, "severe bradycardia", True),        # Severe bradycardia (<40 BPM)
            (0.1, 90, 50, "atrial fibrillation", True),        # Simulated atrial fibrillation (highly irregular)
            (0.05, 120, 10, "ventricular tachycardia", True),  # Ventricular tachycardia (>180 BPM, low variation)
            (0.3, 45, 45, "high variability", True),           # Abnormally high heart rate variability
            (0.4, 20, 1, "very low variability", True),        # Abnormally low heart rate variability
            (0.25, 0, 40, "bigeminy pattern", True)            # Alternating high-low pattern (bigeminy)
        ]
        
        # Combine patterns based on configuration
        patterns = normal_patterns
        if self.include_abnormal:
            patterns = normal_patterns + abnormal_patterns
        
        # Time tracking for pattern changes
        current_pattern = 0
        pattern_time_left = patterns[current_pattern][0] * 60 * 60  # Convert hours to seconds
        
        # For natural trends (gradual changes)
        current_hr_offset = 0
        target_hr_offset = 0
        transition_steps = 0
        steps_remaining = 0
        
        # For pattern-specific variables
        bigeminy_high = True  # For alternating patterns
        skip_beat_counter = 0  # For skipped beats
        
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
                    
                    # Reset pattern-specific variables
                    bigeminy_high = True
                    skip_beat_counter = 0
                    
                    # alert_indicator = "⚠️ ABNORMAL" if patterns[current_pattern][4] else "NORMAL"
                    # print(f"[Heart Rate Sensor] Switching to {patterns[current_pattern][3]} pattern - {alert_indicator}")
                
                # Handle smooth transitions between patterns
                if steps_remaining > 0:
                    current_hr_offset += (target_hr_offset - current_hr_offset) / steps_remaining
                    steps_remaining -= 1
                
                # Get current pattern details
                hr_change = current_hr_offset
                hr_variation = patterns[current_pattern][2]
                pattern_name = patterns[current_pattern][3]
                
                # Handle special pattern types
                if pattern_name == "atrial fibrillation":
                    # Highly irregular heart rate with occasional very fast beats
                    natural_variation = random.uniform(-hr_variation, hr_variation)
                    # Add occasional very rapid beats
                    if random.random() < 0.2:  # 20% chance of a rapid beat
                        natural_variation += random.uniform(20, 40)
                    random_variation = random.uniform(-10, 30)
                
                elif pattern_name == "bigeminy pattern":
                    # Alternating high-low pattern
                    bigeminy_high = not bigeminy_high
                    natural_variation = 30 if bigeminy_high else -15
                    random_variation = random.uniform(-5, 5)
                
                elif pattern_name == "ventricular tachycardia":
                    # Very fast, relatively regular rhythm
                    natural_variation = random.uniform(-hr_variation/2, hr_variation/2)
                    random_variation = random.uniform(0, 10)  # Small upward bias
                
                elif pattern_name == "very low variability":
                    # Almost no variation (concerning for cardiac autonomic neuropathy)
                    natural_variation = random.uniform(-hr_variation/2, hr_variation/2)
                    random_variation = 0
                
                else:
                    # Standard pattern with sinusoidal component for natural rhythm
                    time_factor = time.time() / 30  # Slow oscillation
                    natural_variation = math.sin(time_factor) * (hr_variation / 2)
                    random_variation = random.uniform(-hr_variation/2, hr_variation/2)
                
                # Calculate final heart rate with realistic constraints
                heart_rate = round(base_heart_rate + hr_change + natural_variation + random_variation)
                
                # Handle skipped beats / pauses (occurs in cardiac arrhythmias)
                skip_beat = False
                if pattern_name in ["atrial fibrillation", "cardiac anomaly"] and random.random() < 0.05:
                    skip_beat_counter += 1
                    if skip_beat_counter % 3 == 0:  # Every 3rd potential skip actually happens
                        skip_beat = True
                        heart_rate = max(30, heart_rate // 2)  # Dramatic drop
                
                # Ensure values are in reasonable ranges, but allow extremes for abnormal patterns
                if pattern_name in ["ventricular tachycardia"]:
                    heart_rate = max(140, min(250, heart_rate))  # Allow very high rates
                elif pattern_name in ["severe bradycardia"]:
                    heart_rate = max(30, min(45, heart_rate))    # Allow very low rates
                else:
                    heart_rate = max(40, min(200, heart_rate))   # Standard range
                
                # Write data to CSV
                with open(config.HEART_RATE_CSV, 'a', newline='') as csvfile:
                    fieldnames = ['timestamp', 'heart_rate']
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writerow({
                        'timestamp': timestamp,
                        'heart_rate': heart_rate
                    })
                
                # Wait before next reading
                time.sleep(self.frequency)
                
        except Exception as e:
            print(f"[Heart Rate Sensor] Error: {str(e)}")
            self.running = False

# For backwards compatibility and direct script execution
def heart_rate_sensor(frequency=10, include_abnormal=True):
    """Legacy function to start the heart rate sensor with default frequency."""
    sensor = HeartRateSensor(frequency=frequency, include_abnormal=include_abnormal)
    sensor.start()
    try:
        # Keep the main thread alive
        while sensor.running:
            time.sleep(1)
    except KeyboardInterrupt:
        sensor.stop()