import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import time
import os
import sys 

current_dir = os.path.dirname(os.path.abspath(__file__))
config_dir = os.path.join(current_dir, "../../..")
sys.path.insert(0, config_dir)
from config import DATA_STORAGE_PATH

class parkCamPosture:
    def __init__(self, monitoring_interval=30, pixel_to_cm=0.5, trail_length=30,
                 camera_index=0, data_dir=f"{DATA_STORAGE_PATH}/postural_data",
                 min_detection_confidence=0.5, min_tracking_confidence=0.5):
        """
        Parameters:
          monitoring_interval: Time in seconds for each data segment (default 60 seconds)
          pixel_to_cm: Conversion factor from pixel distance to centimeters
          trail_length: Number of previous CoM points to draw in the trajectory
          camera_index: Video capture device index
          data_dir: Directory for saving CSV files and plots
        """
        self.monitoring_interval = monitoring_interval
        self.pixel_to_cm = pixel_to_cm
        self.trail_length = trail_length
        self.camera_index = camera_index
        self.data_dir = data_dir
        os.makedirs(self.data_dir, exist_ok=True)

        # Initialize MediaPipe Pose model and drawing utilities
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(min_detection_confidence=min_detection_confidence,
                                      min_tracking_confidence=min_tracking_confidence)
        self.mp_draw = mp.solutions.drawing_utils

        # Initialize tracking data for the current interval
        self.reset_interval_data()
        self.interval_start_time = time.time()

    def reset_interval_data(self):
        """Reset the sway positions and timestamp lists for a new monitoring interval."""
        self.sway_positions = []  # Each entry is (com_x, com_y)
        self.time_stamps = []     # Relative timestamps (in seconds)

    def process_interval(self):
        """Compute sway metrics and save the data and plot for the current interval."""
        if len(self.sway_positions) < 2:
            print("Not enough data to process this interval.")
            return

        # Compute sway distances and velocities between consecutive CoM positions.
        sway_distances = []
        sway_velocity = []
        prev_time = self.time_stamps[0]
        prev_position = np.array(self.sway_positions[0])
        for i in range(1, len(self.sway_positions)):
            current_position = np.array(self.sway_positions[i])
            # Convert pixel displacement to centimeters
            displacement = np.linalg.norm(current_position - prev_position) * self.pixel_to_cm
            sway_distances.append(displacement)
            time_diff = self.time_stamps[i] - prev_time
            sway_velocity.append(displacement / time_diff if time_diff > 0 else 0)
            prev_time = self.time_stamps[i]
            prev_position = current_position

        # Calculate the Postural Tremor Index as the standard deviation of the sway distances
        postural_tremor_index = np.std(sway_distances)

        # Because sway_distances and sway_velocity have one fewer element than positions,
        # use the data starting at the second recorded point.
        min_length = min(len(sway_distances), len(sway_velocity), len(self.sway_positions[1:]), len(self.time_stamps[1:]))

        df = pd.DataFrame({
            "Time (s)": self.time_stamps[1:1+min_length],
            "X": [p[0] for p in self.sway_positions[1:1+min_length]],
            "Y": [p[1] for p in self.sway_positions[1:1+min_length]],
            "Sway Distance (cm)": sway_distances[:min_length],
            "Sway Velocity (cm/s)": sway_velocity[:min_length]
        })

        # Save CSV file with current timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        csv_filename = os.path.join(self.data_dir, f"postural_stability_data.csv")
        # Check if the file exists to decide whether to write the header
        write_header = not os.path.exists(csv_filename)
        # Append data to the CSV file
        df.to_csv(csv_filename, mode="a", index=False, header=write_header)
        print(f"Data saved to {csv_filename}")

        # Create a plot with three subplots: Sway Distance, Sway Velocity, and Tremor Index
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 3, 1)
        plt.plot(range(len(sway_distances)), sway_distances, label="Sway Distance", color='b')
        plt.xlabel("Time Step")
        plt.ylabel("Sway Distance (cm)")
        plt.title("Sway Over Time")
        plt.legend()

        plt.subplot(1, 3, 2)
        plt.plot(range(len(sway_velocity)), sway_velocity, label="Sway Velocity", color='r')
        plt.xlabel("Time Step")
        plt.ylabel("Velocity (cm/s)")
        plt.title("Sway Velocity Over Time")
        plt.legend()

        plt.subplot(1, 3, 3)
        plt.bar(["Postural Tremor Index"], [postural_tremor_index], color='g')
        plt.ylabel("Std Dev (cm)")
        plt.title("Tremor in Postural Stability")

        plt.tight_layout()
        plot_filename = os.path.join(self.data_dir, f"postural_stability_plot_{timestamp}.png")
        plt.savefig(plot_filename)
        plt.close()
        print(f"Plot saved to: {plot_filename}")

    def run(self):
        """Run continuous postural stability monitoring. Processes and saves data every fixed interval."""
        cap = cv2.VideoCapture(self.camera_index)
        overall_start_time = time.time()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            current_time = time.time()
            overall_elapsed = current_time - overall_start_time
            interval_elapsed = current_time - self.interval_start_time

            # Process frame with MediaPipe Pose
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(frame_rgb)

            if results.pose_landmarks:
                # Draw the skeleton overlay
                self.mp_draw.draw_landmarks(frame, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)

                # Compute the approximate center of mass (CoM) using the hips
                landmarks = results.pose_landmarks.landmark
                left_hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP]
                right_hip = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP]
                h, w, _ = frame.shape
                com_x = int(((left_hip.x + right_hip.x) / 2) * w)
                com_y = int(((left_hip.y + right_hip.y) / 2) * h)

                # Record the CoM position and timestamp (using overall elapsed time)
                self.sway_positions.append((com_x, com_y))
                self.time_stamps.append(overall_elapsed)

                # Draw the CoM marker
                cv2.circle(frame, (com_x, com_y), 8, (0, 0, 255), -1)

                # Draw the sway trajectory (using the most recent trail_length points)
                start_idx = max(0, len(self.sway_positions) - self.trail_length)
                for i in range(start_idx, len(self.sway_positions) - 1):
                    cv2.line(frame, self.sway_positions[i], self.sway_positions[i + 1], (255, 0, 0), 2)

            # Display elapsed time and an instruction message
            cv2.putText(frame, f"Time: {overall_elapsed:.1f}s", (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            cv2.putText(frame, "Maintain Balance", (20, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            cv2.imshow("Postural Stability Test", frame)

            # If the monitoring interval has elapsed, process and save the current interval's data
            if interval_elapsed >= self.monitoring_interval:
                self.process_interval()
                self.reset_interval_data()
                self.interval_start_time = time.time()  # Restart interval timing

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        # Optionally, process any remaining data if available
        if self.sway_positions:
            self.process_interval()

def main():
    # Create an instance with a monitoring interval of 60 seconds (modifiable)
    monitor = parkCamPosture(monitoring_interval=30)
    monitor.run()

if __name__ == "__main__":
    main()
