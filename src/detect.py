# Project: Player Re-ID (Single-Feed) - Day 1 Setup & YOLOv11 Detection

"""
This file contains the initial project structure and a starter detect.py script to
run YOLOv11 player detection on sample frames of the 15-second video.

Steps we cover:
 1. Directory structure
 2. Virtual environment & dependencies
 3. Download assets
 4. detect.py skeleton with detailed comments
"""

# 1. Directory Structure - create these folders in your project root:
# 
# player-reid/
# ├── data/               # place input videos here
# │    └── 15sec_input_720p.mp4
# ├── weights/            # YOLO model weights go here
# │    └── yolov11.pt
# ├── detections/         # output images with drawn detections
# ├── src/                # all Python code
# │    ├── detect.py      # we will write this today
# │    └── utils.py       # helper functions (empty for now)
# ├── README.md           # to document setup & running
# └── report.md           # initial notes for methodology/report

# 2. Virtual Environment & Dependencies
# In your terminal, run:
#    python -m venv venv
#    venv\Scripts\activate         # Windows
#    pip install --upgrade pip
#    pip install torch torchvision opencv-python ultralytics numpy
#
# Why:
# - venv isolates dependencies
# - torch & torchvision for model backend
# - ultralytics gives YOLOv11 interface
# - OpenCV for video I/O and drawing
# - NumPy for array operations

# 3. Download Assets
# - Download 15sec_input_720p.mp4 → place under data/
# - Download yolov11.pt weights → place under weights/

# 4. src/detect.py Skeleton

import cv2
import numpy as np
from ultralytics import YOLO
import os

# -- Configuration --
VIDEO_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', '15sec_input_720p.mp4')
WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), '..', 'weights', 'yolov11.pt')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'detections')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -- Load Model --
# We use YOLOv11 to detect only the "person" (player) class.
model = YOLO(WEIGHTS_PATH)

# -- Helper: Draw boxes on image --
def draw_boxes(image, detections):
    """
    Draws bounding boxes and confidence on the image.
    detections: list of [x1, y1, x2, y2, conf]
    """
    for x1, y1, x2, y2, conf in detections:
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        label = f"{conf:.2f}"
        cv2.putText(image, label, (int(x1), int(y1)-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
    return image

# -- Main Detection Loop --
def main():
    cap = cv2.VideoCapture(VIDEO_PATH)
    frame_idx = 0
    print("Starting detection on video:", VIDEO_PATH)

    while frame_idx < 10:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)[0]

        detections = []
        for box in results.boxes:
            cls = int(box.cls)
            if cls != 0:
                continue
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = box.conf[0].item()
            detections.append([x1, y1, x2, y2, conf])

        out = draw_boxes(frame.copy(), detections)
        out_path = os.path.join(OUTPUT_DIR, f"det_{frame_idx:03d}.jpg")
        cv2.imwrite(out_path, out)

        frame_idx += 1


    cap.release()
    print(f"Saved detection samples to {OUTPUT_DIR}")

if __name__ == '__main__':
    main()
