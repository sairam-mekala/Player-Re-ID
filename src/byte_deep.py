import os
import cv2
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
# If you’re also using ByteTrack, import it here:
# from yolox.tracker.byte_tracker import BYTETracker

# ─── FIXED PATHS ──────────────────────────────────────────────────────────────
# __file__ is src/byte_deep.py, so BASE_DIR = the src/ folder.
BASE_DIR    = os.path.dirname(__file__)
# Go up one level to the project root:
ROOT_DIR    = os.path.abspath(os.path.join(BASE_DIR, os.pardir))

# Now point to the correct locations:
VIDEO_PATH     = os.path.join(ROOT_DIR, 'data', '15sec_input_720p.mp4')
WEIGHTS_PATH   = os.path.join(ROOT_DIR, 'weights', 'yolov11.pt')
OUTPUT_DIR     = os.path.join(ROOT_DIR, 'detections')
os.makedirs(OUTPUT_DIR, exist_ok=True)
# ───────────────────────────────────────────────────────────────────────────────

# Load YOLO
print("Loading YOLOv11 model from:", WEIGHTS_PATH)
detector = YOLO(WEIGHTS_PATH)
print("Model class names:", detector.names)

# Initialize DeepSORT
tracker = DeepSort(
    max_age=5,
    n_init=5,
    nms_max_overlap=1.0,
    max_cosine_distance=0.2,
)

# (If you were using ByteTrack in parallel, also init it here.)

def draw_tracks(frame, tracks):
    for track in tracks:
        if not track.is_confirmed():
            continue
        x, y, w, h = track.to_ltwh()
        tid = track.track_id
        cv2.rectangle(frame, (int(x), int(y)), (int(x+w), int(y+h)), (255,0,0), 2)
        cv2.putText(frame, f"ID {tid}", (int(x), int(y)-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)
    return frame

if __name__ == '__main__':
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {VIDEO_PATH}")

    frame_idx = 0
    print("Starting tracking…")

    while frame_idx < 200:
        ret, frame = cap.read()
        if not ret:
            break

        # YOLO detection
        res = detector.predict(source=frame, conf=0.5, iou=0.6)[0]
        detections = []
        for box in res.boxes:
            cls  = int(box.cls)
            conf = float(box.conf)
            if detector.names[cls] != 'player' or conf < 0.5:
                continue
            x1,y1,x2,y2 = box.xyxy[0].tolist()
            w, h       = x2 - x1, y2 - y1
            detections.append(([x1,y1,w,h], conf, 'player'))

        # DeepSORT update
        tracks = tracker.update_tracks(detections, frame=frame)

        # Draw & save
        out = draw_tracks(frame.copy(), tracks)
        cv2.imwrite(os.path.join(OUTPUT_DIR, f"frame_{frame_idx:04d}.jpg"), out)

        frame_idx += 1

    cap.release()
    print(f"Done; saved {frame_idx} frames to {OUTPUT_DIR}")
