import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import time
import os

class parkCamTremor:
    def __init__(self, tremor_duration=30, camera_index=0,
                 detection_confidence=0.5, tracking_confidence=0.5,
                 data_dir="data_storage/tremor_data"):
        """
        Parameters:
          tremor_duration: Test duration in seconds.
          camera_index: Which video capture device to use.
          detection_confidence: Minimum detection confidence for MediaPipe.
          tracking_confidence: Minimum tracking confidence for MediaPipe.
          data_dir: Directory where CSV and plot files will be saved.
        """
        self.tremor_duration = tremor_duration
        self.camera_index = camera_index
        self.detection_confidence = detection_confidence
        self.tracking_confidence = tracking_confidence
        self.data_dir = data_dir

        # Create data directory if it doesn't exist
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)

        # Initialize MediaPipe Hands model and drawing utilities
        self.mp_hands = mp.solutions.hands
        self.hands_model = self.mp_hands.Hands(
            min_detection_confidence=self.detection_confidence,
            min_tracking_confidence=self.tracking_confidence
        )
        self.mp_draw = mp.solutions.drawing_utils

        # Containers for wrist positions and timestamps
        self.hand_positions = {"left": [], "right": []}
        self.time_stamps = {"left": [], "right": []}

    def run(self):
        """Run the resting tremor test for the specified duration."""
        cap = cv2.VideoCapture(self.camera_index)
        start_time = time.time()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            current_time = time.time()
            elapsed_time = current_time - start_time
            if elapsed_time > self.tremor_duration:
                break

            # Process frame for MediaPipe Hands
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands_model.process(frame_rgb)

            if results.multi_hand_landmarks:
                for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                    # Determine hand label
                    hand_label = "left" if handedness.classification[0].label.lower() == "left" else "right"
                    # Get wrist landmark and convert to pixel coordinates
                    wrist = hand_landmarks.landmark[self.mp_hands.HandLandmark.WRIST]
                    h, w, _ = frame.shape
                    wrist_x, wrist_y = int(wrist.x * w), int(wrist.y * h)

                    # Store data for this hand
                    self.hand_positions[hand_label].append((wrist_x, wrist_y))
                    self.time_stamps[hand_label].append(current_time)

                    # Draw landmarks and wrist marker on frame
                    self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                    cv2.circle(frame, (wrist_x, wrist_y), 5, (0, 0, 255), -1)

            # Overlay elapsed time and instruction text
            cv2.putText(frame, f"Time: {elapsed_time:.1f}s / {self.tremor_duration}s", (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            cv2.putText(frame, "Keep hands at rest", (20, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            cv2.imshow("Resting Tremor Detection", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    def compute_metrics(self):
        """
        Process the collected wrist positions to compute tremor metrics.
        Returns:
          tremor_data: Dictionary with average frequency, amplitude, and tremor index for each hand.
          csv_data: Tuple of lists for CSV output (times, hand labels, frequencies, amplitudes).
          asymmetry_score: Absolute frequency difference between left and right hands (if both exist).
        """
        tremor_data = {}
        tremor_amplitudes_all = []
        tremor_frequencies_all = []
        hands_all = []
        times_all = []

        for hand in ["left", "right"]:
            if len(self.hand_positions[hand]) > 1:
                tremor_amplitudes = []
                tremor_frequencies = []
                for i in range(1, len(self.hand_positions[hand])):
                    prev_position = np.array(self.hand_positions[hand][i - 1])
                    current_position = np.array(self.hand_positions[hand][i])
                    displacement = np.linalg.norm(current_position - prev_position)
                    time_diff = self.time_stamps[hand][i] - self.time_stamps[hand][i - 1]
                    tremor_frequency = 1 / time_diff if time_diff > 0 else 0

                    tremor_amplitudes.append(displacement)
                    tremor_frequencies.append(tremor_frequency)

                    # Collect data for CSV output
                    tremor_amplitudes_all.append(displacement)
                    tremor_frequencies_all.append(tremor_frequency)
                    hands_all.append(hand)
                    times_all.append(self.time_stamps[hand][i])

                tremor_index = np.std(tremor_amplitudes)
                avg_tremor_freq = np.mean(tremor_frequencies)
                avg_tremor_amp = np.mean(tremor_amplitudes)
                tremor_data[hand] = {
                    "frequency": avg_tremor_freq,
                    "amplitude": avg_tremor_amp,
                    "tremor_index": tremor_index
                }

        asymmetry_score = None
        if "left" in tremor_data and "right" in tremor_data:
            asymmetry_score = abs(tremor_data["left"]["frequency"] - tremor_data["right"]["frequency"])

        return tremor_data, (times_all, hands_all, tremor_frequencies_all, tremor_amplitudes_all), asymmetry_score

    def process_and_save(self):
        """
        Computes tremor metrics, saves the data to a CSV file (within the data folder),
        and plots the tremor frequency, amplitude, and asymmetry.
        """
        tremor_data, csv_data, asymmetry_score = self.compute_metrics()
        times_all, hands_all, tremor_frequencies_all, tremor_amplitudes_all = csv_data

        # Save the collected data to CSV in the specified data folder
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        csv_filename = os.path.join(self.data_dir, f"resting_tremor_data_{timestamp}.csv")
        df = pd.DataFrame({
            "Time (s)": times_all,
            "Hand": hands_all,
            "Tremor Frequency (Hz)": tremor_frequencies_all,
            "Tremor Amplitude (px)": tremor_amplitudes_all
        })
        df.to_csv(csv_filename, index=False)
        print(f"Data saved to {csv_filename}")

        # Plot the tremor metrics if data exists
        plt.figure(figsize=(12, 5))

        # Tremor Frequency Plot
        if "left" in tremor_data or "right" in tremor_data:
            labels, freqs = [], []
            if "left" in tremor_data:
                labels.append("Left Hand")
                freqs.append(tremor_data["left"]["frequency"])
            if "right" in tremor_data:
                labels.append("Right Hand")
                freqs.append(tremor_data["right"]["frequency"])
            plt.subplot(1, 3, 1)
            plt.bar(labels, freqs, color=['blue', 'orange'])
            plt.ylabel("Frequency (Hz)")
            plt.title("Tremor Frequency")

        # Tremor Amplitude Plot
        if "left" in tremor_data or "right" in tremor_data:
            labels, amps = [], []
            if "left" in tremor_data:
                labels.append("Left Hand")
                amps.append(tremor_data["left"]["amplitude"])
            if "right" in tremor_data:
                labels.append("Right Hand")
                amps.append(tremor_data["right"]["amplitude"])
            plt.subplot(1, 3, 2)
            plt.bar(labels, amps, color=['blue', 'orange'])
            plt.ylabel("Amplitude (px)")
            plt.title("Tremor Amplitude")

        # Tremor Asymmetry Plot (if both hands are present)
        if asymmetry_score is not None:
            plt.subplot(1, 3, 3)
            plt.bar(["Asymmetry"], [asymmetry_score], color='red')
            plt.ylabel("Hz Difference")
            plt.title("Tremor Asymmetry")

        plt.tight_layout()
        plt.show()

def main():
    # Create and run the resting tremor detector for 30 seconds,
    # with data saved in the "tremor_results" folder.
    detector = parkCamTremor(tremor_duration=30)
    detector.run()

    # Compute metrics, save data to CSV, and plot results (all within the class)
    detector.process_and_save()

if __name__ == "__main__":
    main()
