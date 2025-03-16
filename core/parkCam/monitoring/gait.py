import cv2
import mediapipe as mp
import numpy as np
import time
import csv
import os
import matplotlib.pyplot as plt
import sys 

current_dir = os.path.dirname(os.path.abspath(__file__))
config_dir = os.path.join(current_dir, "../../..")
sys.path.insert(0, config_dir)
from config import DATA_STORAGE_PATH

class parkCamGait:
    def __init__(self, interval_duration=30, data_dir="{DATA_STORAGE_PATH}/gait_data", frame_rate=30,
                 step_threshold=0.05, shuffling_threshold=0.02, camera_index=0):
        """
        Parameters:
          interval_duration: time (in seconds) for each data segment (default: 60 seconds)
          data_dir: directory where CSV and plot files will be saved
          frame_rate: expected frame rate (for reference)
          step_threshold: minimum horizontal distance change to count a step
          shuffling_threshold: threshold on vertical change to indicate shuffling gait
          camera_index: which video capture device to use
        """
        self.interval_duration = interval_duration
        self.data_dir = data_dir
        self.frame_rate = frame_rate
        self.step_threshold = step_threshold
        self.shuffling_threshold = shuffling_threshold
        self.camera_index = camera_index

        # Create directory for saving data
        os.makedirs(self.data_dir, exist_ok=True)

        # Initialize MediaPipe Pose
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose()
        self.mp_draw = mp.solutions.drawing_utils

        # Initialize/reset tracking data for the current interval
        self.reset_interval_data()

    def reset_interval_data(self):
        """Reset tracking data for a new interval."""
        self.left_foot_y = []
        self.right_foot_y = []
        self.left_foot_x = []
        self.right_foot_x = []
        self.timestamps = []
        self.step_count = 0
        self.shuffling_count = 0
        self.interval_start_time = time.time()

    def process_interval(self):
        """Compute gait metrics and save CSV files and plot for the current interval."""
        if not self.timestamps:
            total_time = self.interval_duration
        else:
            total_time = self.timestamps[-1] - self.timestamps[0]
            if total_time <= 0:
                total_time = 1

        # Calculate cadence (steps per minute)
        cadence = (self.step_count / total_time) * 60

        # Compute stride symmetry ratio (if possible)
        if len(self.left_foot_x) > 2 and len(self.right_foot_x) > 2:
            left_diff = np.diff(self.left_foot_x)
            right_diff = np.diff(self.right_foot_x)
            if np.std(right_diff) != 0:
                stride_symmetry = np.std(left_diff) / np.std(right_diff)
            else:
                stride_symmetry = 0
        else:
            stride_symmetry = 0

        # Save gait metrics to CSV
        # metrics_filename = os.path.join(self.data_dir, f"gait_metrics_{int(self.interval_start_time)}.csv")
        metrics_filename = os.path.join(self.data_dir, f"gait_metrics.csv")
        file_exists = os.path.isfile(metrics_filename) and os.path.getsize(metrics_filename) > 0

        with open(metrics_filename, mode="a", newline="") as file:
            writer = csv.writer(file)
            if not file_exists:
                writer.writerow(["Total Steps", "Cadence (steps/min)", "Stride Symmetry Ratio", "Shuffling Count"])
            writer.writerow([self.step_count, cadence, stride_symmetry, self.shuffling_count])
        print(f"Gait metrics saved to: {metrics_filename}")

        # Save foot trajectory data to CSV
        # trajectory_filename = os.path.join(self.data_dir, f"foot_trajectory_{int(self.interval_start_time)}.csv")
        trajectory_filename = os.path.join(self.data_dir, f"foot_trajectory.csv")
        file_exists = os.path.isfile(trajectory_filename) and os.path.getsize(trajectory_filename) > 0
        
        with open(trajectory_filename, mode="a", newline="") as file:
            writer = csv.writer(file)
            if not file_exists:
                writer.writerow(["Time (s)", "Left Foot X", "Left Foot Y", "Right Foot X", "Right Foot Y"])
            for i in range(len(self.timestamps)):
                writer.writerow([self.timestamps[i], self.left_foot_x[i], self.left_foot_y[i],
                                 self.right_foot_x[i], self.right_foot_y[i]])
        print(f"Foot trajectory data saved to: {trajectory_filename}")

        # Save a plot of foot trajectories for this interval
        plt.figure(figsize=(10, 5))
        plt.plot(self.timestamps, self.left_foot_y, label="Left Foot Height", color="blue")
        plt.plot(self.timestamps, self.right_foot_y, label="Right Foot Height", color="red")
        plt.xlabel("Time (s)")
        plt.ylabel("Foot Height")
        plt.title("Gait Analysis - Foot Trajectory")
        plt.legend()
        plot_filename = os.path.join(self.data_dir, f"foot_trajectory_{int(self.interval_start_time)}.png")
        plt.savefig(plot_filename)
        plt.close()
        print(f"Foot trajectory plot saved to: {plot_filename}")

    def run(self):
        """Start the continuous gait monitoring. Data is processed and saved every interval."""
        cap = cv2.VideoCapture(self.camera_index)
        overall_start_time = time.time()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Convert the frame to RGB and process pose landmarks
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(image)

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                left_heel = landmarks[self.mp_pose.PoseLandmark.LEFT_HEEL]
                right_heel = landmarks[self.mp_pose.PoseLandmark.RIGHT_HEEL]

                # Record current timestamp relative to overall start
                current_time = time.time() - overall_start_time
                self.timestamps.append(current_time)
                self.left_foot_x.append(left_heel.x)
                self.left_foot_y.append(left_heel.y)
                self.right_foot_x.append(right_heel.x)
                self.right_foot_y.append(right_heel.y)

                # Step detection (horizontal movement)
                if len(self.left_foot_x) > 1 and len(self.right_foot_x) > 1:
                    left_step = abs(self.left_foot_x[-1] - self.left_foot_x[-2]) > self.step_threshold
                    right_step = abs(self.right_foot_x[-1] - self.right_foot_x[-2]) > self.step_threshold
                    if left_step or right_step:
                        self.step_count += 1

                # Shuffling detection (low vertical movement)
                if len(self.left_foot_y) > 1 and len(self.right_foot_y) > 1:
                    if abs(self.left_foot_y[-1] - self.left_foot_y[-2]) < self.shuffling_threshold:
                        self.shuffling_count += 1
                    if abs(self.right_foot_y[-1] - self.right_foot_y[-2]) < self.shuffling_threshold:
                        self.shuffling_count += 1

                # Draw pose landmarks
                self.mp_draw.draw_landmarks(frame, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)

            # Display current step count and shuffling count
            cv2.putText(frame, f"Steps: {self.step_count}", (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Shuffling Count: {self.shuffling_count}", (30, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow('Gait Analysis', frame)

            # Check if the current interval has elapsed
            if time.time() - self.interval_start_time >= self.interval_duration:
                self.process_interval()
                self.reset_interval_data()  # start a new interval

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

def main():
    # Set monitoring interval to 60 seconds (or any desired duration)
    monitor = parkCamGait(interval_duration=30)
    monitor.run()

if __name__ == "__main__":
    main()
