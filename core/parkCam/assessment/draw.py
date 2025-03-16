import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import time

import os, sys

current_dir = os.path.dirname(os.path.abspath(__file__))
config_dir = os.path.join(current_dir, "../../..")
sys.path.insert(0, config_dir)
from config import DATA_STORAGE_PATH

class parkCamDraw:
    def __init__(self, spiral_radius=200, center=(250, 250), step_size=5, sampling_rate=0.02):
        self.spiral_radius = spiral_radius
        self.center = center
        self.step_size = step_size
        self.sampling_rate = sampling_rate
        
        # Data storage for drawn points and timestamps
        self.drawing_points = []
        self.time_stamps = []
        
        # Pre-calculate the reference (ideal) spiral points
        self.ideal_spiral = self.generate_spiral()

    def generate_spiral(self):
        """Generate reference spiral points."""
        theta = np.arange(0, 5 * np.pi, 0.1)  # 5 full turns
        spiral_x = self.center[0] + self.spiral_radius * np.cos(theta) * (theta / max(theta))
        spiral_y = self.center[1] + self.spiral_radius * np.sin(theta) * (theta / max(theta))
        return list(zip(spiral_x, spiral_y))

    def mouse_callback(self, event, x, y, flags, param):
        """Capture mouse movements while the left button is held down."""
        if event == cv2.EVENT_MOUSEMOVE and flags == cv2.EVENT_FLAG_LBUTTON:
            self.drawing_points.append((x, y))
            self.time_stamps.append(time.time())

    def run_drawing(self):
        """Display the canvas, handle mouse input, and capture the drawing."""
        # Create a white canvas and draw the ideal spiral points
        canvas = np.ones((500, 500, 3), dtype=np.uint8) * 255
        for point in self.ideal_spiral:
            cv2.circle(canvas, (int(point[0]), int(point[1])), 1, (200, 200, 200), -1)

        cv2.imshow("Spiral Drawing Test", canvas)
        cv2.setMouseCallback("Spiral Drawing Test", self.mouse_callback)

        # Main drawing loop
        while True:
            temp_canvas = canvas.copy()
            # Draw the user-drawn spiral
            for i in range(len(self.drawing_points) - 1):
                cv2.line(temp_canvas, self.drawing_points[i], self.drawing_points[i + 1], (0, 0, 255), 2)
            cv2.imshow("Spiral Drawing Test", temp_canvas)

            # Exit on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()
        return self.drawing_points, self.time_stamps

    def compute_metrics(self, drawing_points, time_stamps):
        """Compute deviation from the ideal spiral and drawing velocity."""
        if len(drawing_points) == 0:
            return None, None, None

        deviations = []
        velocities = []
        prev_time = time_stamps[0]

        for i, (x, y) in enumerate(drawing_points):
            # Find the closest reference spiral point
            closest_point = min(self.ideal_spiral, key=lambda p: np.linalg.norm(np.array(p) - np.array([x, y])))
            deviation = np.linalg.norm(np.array(closest_point) - np.array([x, y]))
            deviations.append(deviation)

            # Compute velocity (distance / time)
            if i > 0:
                dist = np.linalg.norm(np.array(drawing_points[i]) - np.array(drawing_points[i - 1]))
                time_diff = time_stamps[i] - prev_time
                prev_time = time_stamps[i]
                velocities.append(dist / time_diff if time_diff > 0 else 0)

        # Tremor index: standard deviation of deviations
        tremor_index = np.std(deviations)
        return deviations, velocities, tremor_index

def main():
    # Instantiate the drawing class and run the drawing capture
    drawer = parkCamDraw()
    drawing_points, time_stamps = drawer.run_drawing()

    if drawing_points:
        deviations, velocities, tremor_index = drawer.compute_metrics(drawing_points, time_stamps)

        # Save data to CSV
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        df = pd.DataFrame({
            "Time (s)": time_stamps[:len(deviations)],
            "X": [p[0] for p in drawing_points[:len(deviations)]],
            "Y": [p[1] for p in drawing_points[:len(deviations)]],
            "Deviation (pixels)": deviations,
            "Velocity (px/s)": velocities + [None]  # Last velocity entry is None to match length
        })
        csv_filename = f"{DATA_STORAGE_PATH}/spiral_drawing_data_{timestamp}.csv"
        df.to_csv(csv_filename, index=False)
        print(f"Data saved to {csv_filename}")

        # Plot results
        plt.figure(figsize=(12, 5))

        # Deviation Plot
        plt.subplot(1, 3, 1)
        plt.plot(range(len(deviations)), deviations, label="Deviation")
        plt.xlabel("Stroke #")
        plt.ylabel("Deviation (pixels)")
        plt.title("Deviation from Ideal Spiral")
        plt.legend()

        # Velocity Plot
        plt.subplot(1, 3, 2)
        plt.plot(range(len(velocities)), velocities, label="Velocity", color='r')
        plt.xlabel("Stroke #")
        plt.ylabel("Velocity (px/s)")
        plt.title("Drawing Speed")
        plt.legend()

        # Tremor Index Plot
        plt.subplot(1, 3, 3)
        plt.bar(["Tremor Index"], [tremor_index], color='g')
        plt.ylabel("Standard Deviation of Deviation")
        plt.title("Tremor Index")

        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    main()
