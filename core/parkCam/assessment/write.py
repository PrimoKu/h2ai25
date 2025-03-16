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

class parkCamWrite:
    def __init__(self, canvas_width=800, canvas_height=600,
                 instruction_text="Write the sentence: 'Today is a good day.'  Press 'n' for new sentence, 'q' to quit."):
        self.canvas_width = canvas_width
        self.canvas_height = canvas_height
        self.instruction_text = instruction_text
        self.window_name = "Handwriting Task"
        
        # Initialize the canvas and add instructions
        self.canvas = np.ones((self.canvas_height, self.canvas_width, 3), dtype=np.uint8) * 255
        cv2.putText(self.canvas, self.instruction_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        
        # Variables to track strokes and sentences
        self.current_stroke = []    # Stores the current stroke (list of (x, y, timestamp))
        self.current_sentence = []  # List of strokes for the current sentence
        self.sentences_data = []    # Metrics for each completed sentence
        
        # Set up the OpenCV window and mouse callback
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.draw_callback)

    def draw_callback(self, event, x, y, flags, param):
        """Mouse callback to capture handwriting strokes."""
        if event == cv2.EVENT_LBUTTONDOWN:
            # Start a new stroke
            self.current_stroke = [(x, y, time.time())]
        elif event == cv2.EVENT_MOUSEMOVE and flags & cv2.EVENT_FLAG_LBUTTON:
            # Record stroke points as the mouse moves with button held
            self.current_stroke.append((x, y, time.time()))
        elif event == cv2.EVENT_LBUTTONUP:
            # End the stroke and add it to the current sentence
            self.current_stroke.append((x, y, time.time()))
            self.current_sentence.append(self.current_stroke)

    def run(self):
        """Run the handwriting task until the user quits."""
        while True:
            temp_canvas = self.canvas.copy()
            
            # Draw completed strokes from the current sentence
            for stroke in self.current_sentence:
                for i in range(len(stroke) - 1):
                    pt1 = (stroke[i][0], stroke[i][1])
                    pt2 = (stroke[i+1][0], stroke[i+1][1])
                    cv2.line(temp_canvas, pt1, pt2, (0, 0, 255), 2)
            
            # Draw the ongoing stroke (if any)
            if self.current_stroke:
                for i in range(len(self.current_stroke) - 1):
                    pt1 = (self.current_stroke[i][0], self.current_stroke[i][1])
                    pt2 = (self.current_stroke[i+1][0], self.current_stroke[i+1][1])
                    cv2.line(temp_canvas, pt1, pt2, (0, 0, 255), 2)
            
            cv2.imshow(self.window_name, temp_canvas)
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('n'):
                # When 'n' is pressed, complete the current sentence and compute its metrics
                if self.current_sentence:
                    # Flatten all stroke points into one list
                    all_points = [pt for stroke in self.current_sentence for pt in stroke]
                    start_time_sentence = all_points[0][2]
                    end_time_sentence = all_points[-1][2]
                    duration = end_time_sentence - start_time_sentence

                    # Compute total stroke length (sum of distances between consecutive points)
                    total_length = 0
                    for stroke in self.current_sentence:
                        for i in range(len(stroke) - 1):
                            pt1 = np.array(stroke[i][:2])
                            pt2 = np.array(stroke[i+1][:2])
                            total_length += np.linalg.norm(pt2 - pt1)
                    
                    # Estimate letter height using the vertical span of all points
                    ys = [pt[1] for pt in all_points]
                    letter_height = max(ys) - min(ys) if ys else 0

                    sentence_index = len(self.sentences_data) + 1
                    self.sentences_data.append({
                        "Sentence": sentence_index,
                        "Duration (s)": duration,
                        "Total Stroke Length (px)": total_length,
                        "Approx Letter Height (px)": letter_height,
                        "Stroke Count": len(self.current_sentence)
                    })
                    print(f"Sentence {sentence_index}: Duration={duration:.2f}s, Total Length={total_length:.2f}px, "
                          f"Letter Height={letter_height:.2f}px, Stroke Count={len(self.current_sentence)}")
                    
                    # Reset for the next sentence
                    self.current_sentence = []
                    self.current_stroke = []
                    self.canvas = np.ones((self.canvas_height, self.canvas_width, 3), dtype=np.uint8) * 255
                    cv2.putText(self.canvas, self.instruction_text, (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            
            elif key == ord('q'):
                # Quit the task when 'q' is pressed
                break

        cv2.destroyAllWindows()
        return self.sentences_data

def main():
    # Instantiate and run the handwriting task
    task = parkCamWrite()
    sentences_data = task.run()

    # If any sentence was recorded, save the data and plot results
    if sentences_data:
        df = pd.DataFrame(sentences_data)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        csv_filename = f"{DATA_STORAGE_PATH}/handwriting_data_{timestamp}.csv"
        df.to_csv(csv_filename, index=False)
        print(f"Data saved to {csv_filename}")

        # Plot metrics for each sentence
        plt.figure(figsize=(12, 4))

        plt.subplot(1, 3, 1)
        plt.plot(df["Sentence"], df["Duration (s)"], marker='o')
        plt.xlabel("Sentence #")
        plt.ylabel("Duration (s)")
        plt.title("Writing Duration")

        plt.subplot(1, 3, 2)
        plt.plot(df["Sentence"], df["Total Stroke Length (px)"], marker='o', color='r')
        plt.xlabel("Sentence #")
        plt.ylabel("Stroke Length (px)")
        plt.title("Total Stroke Length")

        plt.subplot(1, 3, 3)
        plt.plot(df["Sentence"], df["Approx Letter Height (px)"], marker='o', color='g')
        plt.xlabel("Sentence #")
        plt.ylabel("Approx Letter Height (px)")
        plt.title("Approx Letter Height")

        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    main()
