import cv2
import mediapipe as mp
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime


import os, sys

current_dir = os.path.dirname(os.path.abspath(__file__))
config_dir = os.path.join(current_dir, "../../..")
sys.path.insert(0, config_dir)
from config import DATA_STORAGE_PATH

class parkCamFlip:
    def __init__(self, flip_threshold=20, min_time_between_flips=0.3, 
                 detection_confidence=0.5, tracking_confidence=0.5, camera_index=0):
        # Flip detection parameters
        self.flip_threshold = flip_threshold
        self.min_time_between_flips = min_time_between_flips
        
        # Data storage for flip times and previous wrist angles for each hand
        self.flip_times = {"left": [], "right": []}
        self.prev_angles = {"left": None, "right": None}
        
        # Set up MediaPipe Hands and drawing utilities
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(min_detection_confidence=detection_confidence,
                                          min_tracking_confidence=tracking_confidence)
        # Open video capture
        self.cap = cv2.VideoCapture(camera_index)
        self.start_time = time.time()
        self.elapsed_time = 0

    def calculate_wrist_angle(self, hand_landmarks):
        # Get the wrist, index finger base, and pinky base points
        wrist = np.array([hand_landmarks.landmark[0].x, hand_landmarks.landmark[0].y])
        index_base = np.array([hand_landmarks.landmark[5].x, hand_landmarks.landmark[5].y])
        pinky_base = np.array([hand_landmarks.landmark[17].x, hand_landmarks.landmark[17].y])
        # Calculate the angle between the vector from index base to pinky base and the horizontal
        vector = pinky_base - index_base
        angle = np.arctan2(vector[1], vector[0]) * (180 / np.pi)
        return angle

    def process_frame(self, frame):
        # Convert the BGR frame to RGB and process with MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                # Determine if the detected hand is left or right
                hand_label = "left" if handedness.classification[0].label.lower() == "left" else "right"
                wrist_angle = self.calculate_wrist_angle(hand_landmarks)

                # Detect a flip based on a significant change in wrist angle
                if self.prev_angles[hand_label] is not None:
                    angle_change = abs(wrist_angle - self.prev_angles[hand_label])
                    if angle_change > self.flip_threshold:
                        current_time = time.time()
                        # Avoid double-counting by checking the minimum time between flips
                        if (len(self.flip_times[hand_label]) == 0 or 
                            (current_time - self.flip_times[hand_label][-1]) >= self.min_time_between_flips):
                            self.flip_times[hand_label].append(current_time)
                self.prev_angles[hand_label] = wrist_angle

                # Draw hand landmarks on the frame
                self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
        return frame

    def run(self):
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break

            processed_frame = self.process_frame(frame)

            # Calculate elapsed time and update flip frequencies in real time
            elapsed_time = max(1, time.time() - self.start_time)
            left_freq = len(self.flip_times["left"]) / elapsed_time
            right_freq = len(self.flip_times["right"]) / elapsed_time

            # Display flip counts and frequencies on the frame
            cv2.putText(processed_frame, f"Left Flips: {len(self.flip_times['left'])}", (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(processed_frame, f"Right Flips: {len(self.flip_times['right'])}", (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            cv2.putText(processed_frame, f"Left Freq: {left_freq:.2f} Hz", (20, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            cv2.putText(processed_frame, f"Right Freq: {right_freq:.2f} Hz", (20, 140),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 165, 0), 2)

            cv2.imshow("Hand Flip Task (Pronation-Supination)", processed_frame)

            # Exit when 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Update elapsed time and release resources
        self.elapsed_time = time.time() - self.start_time
        self.cap.release()
        cv2.destroyAllWindows()
        return self.flip_times, self.elapsed_time

def main():
    # Instantiate and run the hand flip detector
    detector = parkCamFlip()
    flip_times, elapsed_time = detector.run()

    # Compute inter-flip intervals for left and right hands
    left_intervals = [None] + list(np.diff(flip_times["left"])) if len(flip_times["left"]) > 0 else []
    right_intervals = [None] + list(np.diff(flip_times["right"])) if len(flip_times["right"]) > 0 else []

    # Compute flip frequencies and asymmetry score
    left_freq = len(flip_times["left"]) / elapsed_time
    right_freq = len(flip_times["right"]) / elapsed_time
    asymmetry = abs(left_freq - right_freq)

    # Create DataFrames for left and right hand flips and concatenate them
    left_data = {
        "Flip Time (s)": flip_times["left"],
        "Hand": ["Left"] * len(flip_times["left"]),
        "Inter-Flip Interval (s)": left_intervals
    }
    right_data = {
        "Flip Time (s)": flip_times["right"],
        "Hand": ["Right"] * len(flip_times["right"]),
        "Inter-Flip Interval (s)": right_intervals
    }
    df_left = pd.DataFrame(left_data)
    df_right = pd.DataFrame(right_data)
    df = pd.concat([df_left, df_right], ignore_index=True)
    
    # Save flip data to a CSV file
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    csv_filename = f"{DATA_STORAGE_PATH}/hand_flip_data_{timestamp}.csv"
    df.to_csv(csv_filename, index=False)
    print(f"Data saved to {csv_filename}")

    # Plot the results
    plt.figure(figsize=(12, 5))

    # Plot flip frequency for each hand
    plt.subplot(1, 3, 1)
    plt.bar(["Left Hand", "Right Hand"], [left_freq, right_freq], color=['blue', 'orange'])
    plt.ylabel("Flip Frequency (Hz)")
    plt.title("Hand Flip Frequency")

    # Plot inter-flip intervals for each hand
    plt.subplot(1, 3, 2)
    plt.plot(range(len(left_intervals)), left_intervals, label="Left Hand", color='blue')
    plt.plot(range(len(right_intervals)), right_intervals, label="Right Hand", color='orange')
    plt.xlabel("Flip #")
    plt.ylabel("Inter-Flip Interval (s)")
    plt.title("Inter-Flip Variability")
    plt.legend()

    # Plot the asymmetry score between left and right hand frequencies
    plt.subplot(1, 3, 3)
    plt.bar(["Asymmetry"], [asymmetry], color='red')
    plt.ylabel("Hz Difference")
    plt.title("Hand Flip Asymmetry")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
