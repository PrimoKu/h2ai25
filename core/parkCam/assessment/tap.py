import cv2
import mediapipe as mp
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import config

class parkCamTap:
    def __init__(self, tap_threshold=30, min_time_between_taps=0.1,
                 detection_confidence=0.5, tracking_confidence=0.5, camera_index=0):
        # Parameters for tap detection
        self.tap_threshold = tap_threshold
        self.min_time_between_taps = min_time_between_taps
        
        # Data storage for tap analysis
        self.tap_times = []
        self.distances = []
        self.prev_distance = None
        
        # Initialize MediaPipe Hands and drawing utilities
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(min_detection_confidence=detection_confidence,
                                         min_tracking_confidence=tracking_confidence)
        # Open video capture
        self.cap = cv2.VideoCapture(camera_index)
        self.start_time = time.time()
        self.elapsed_time = 0

    def run(self):
        """Run the finger tapping detection loop until 'q' is pressed."""
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break

            current_time = time.time()
            # Convert frame to RGB for MediaPipe processing
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(frame_rgb)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Get key points for index finger tip (8) and thumb tip (4)
                    index_finger_tip = hand_landmarks.landmark[8]
                    thumb_tip = hand_landmarks.landmark[4]
                    # Convert landmarks to pixel coordinates
                    h, w, _ = frame.shape
                    index_x, index_y = int(index_finger_tip.x * w), int(index_finger_tip.y * h)
                    thumb_x, thumb_y = int(thumb_tip.x * w), int(thumb_tip.y * h)
                    # Calculate Euclidean distance between index finger and thumb
                    distance = np.sqrt((index_x - thumb_x) ** 2 + (index_y - thumb_y) ** 2)
                    self.distances.append(distance)

                    # Detect tap: a tap occurs when the previous distance is above threshold
                    # and the current distance is below or equal to the threshold.
                    if self.prev_distance is not None and self.prev_distance > self.tap_threshold and distance <= self.tap_threshold:
                        if len(self.tap_times) == 0 or (current_time - self.tap_times[-1]) >= self.min_time_between_taps:
                            self.tap_times.append(current_time)
                    self.prev_distance = distance

                    # Draw hand landmarks and connections on the frame
                    self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

                    # Display real-time tap count, frequency, and amplitude
                    elapsed_time = max(1, current_time - self.start_time)  # avoid division by zero
                    tap_frequency = len(self.tap_times) / elapsed_time
                    cv2.putText(frame, f"Taps: {len(self.tap_times)}", (20, 50), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(frame, f"Freq: {tap_frequency:.2f} Hz", (20, 80), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                    if len(self.distances) > 1:
                        amplitude = np.mean(self.distances)
                        cv2.putText(frame, f"Amplitude: {amplitude:.1f}px", (20, 110), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            # Show the video feed
            cv2.imshow("Finger Tapping Task", frame)
            # Exit on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.elapsed_time = time.time() - self.start_time
        self.cap.release()
        cv2.destroyAllWindows()
        return self.tap_times, self.distances, self.elapsed_time

def main():
    # Instantiate and run the finger tap detector
    detector = parkCamTap()
    tap_times, distances, elapsed_time = detector.run()

    # Compute tap statistics if at least two taps were detected
    if len(tap_times) > 1:
        # Prepend None for the first tap so that the list lengths match
        inter_tap_intervals = [None] + list(np.diff(tap_times))
        # Calculate mean and standard deviation for inter-tap intervals (excluding the None)
        valid_intervals = [it for it in inter_tap_intervals if it is not None]
        mean_interval = np.mean(valid_intervals)
        std_dev_interval = np.std(valid_intervals)
        tap_frequency = len(tap_times) / (tap_times[-1] - tap_times[0])
        avg_amplitude = np.mean(distances[:len(tap_times)])

        print(f"Total Taps: {len(tap_times)}")
        print(f"Tap Frequency: {tap_frequency:.2f} Hz")
        print(f"Mean Tap Interval: {mean_interval:.2f} s")
        print(f"Tap Variability (Std Dev): {std_dev_interval:.2f} s")
        print(f"Average Tap Amplitude: {avg_amplitude:.2f} pixels")

        # Save data to a CSV file
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        df = pd.DataFrame({
            "Tap Times (s)": tap_times,
            "Inter-Tap Interval (s)": inter_tap_intervals,
            "Amplitude (pixels)": distances[:len(tap_times)]
        })
        csv_filename = f"{config.DATA_STORAGE_PATH}/finger_tap_data_{timestamp}.csv"
        df.to_csv(csv_filename, index=False)
        print(f"Data saved to {csv_filename}")

        # Plot results
        plt.figure(figsize=(12, 5))

        # Tap Frequency Plot: constant frequency value over taps
        plt.subplot(1, 3, 1)
        plt.plot(range(len(tap_times)), [tap_frequency] * len(tap_times), label="Frequency")
        plt.xlabel("Tap #")
        plt.ylabel("Frequency (Hz)")
        plt.title("Tap Frequency Over Time")
        plt.legend()

        # Tap Amplitude Plot: amplitude per tap
        plt.subplot(1, 3, 2)
        plt.plot(range(len(tap_times)), distances[:len(tap_times)], label="Amplitude", color='r')
        plt.xlabel("Tap #")
        plt.ylabel("Amplitude (pixels)")
        plt.title("Tap Amplitude Over Time")
        plt.legend()

        # Inter-Tap Variability Plot
        plt.subplot(1, 3, 3)
        # Exclude the first None value for plotting variability
        plt.plot(range(1, len(inter_tap_intervals)), valid_intervals, label="Variability", color='g')
        plt.xlabel("Tap #")
        plt.ylabel("Interval (s)")
        plt.title("Inter-Tap Variability")
        plt.legend()

        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    main()
