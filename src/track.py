import os
import cv2
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

ROOT_DIR = os.path.dirname(__file__)
VIDEO_PATH = os.path.join(ROOT_DIR, '..', 'data', '15sec_input_720p.mp4')
WEIGHTS_PATH = os.path.join(ROOT_DIR, '..', 'weights', 'yolov11.pt')
OUTPUT_DIR = os.path.join(ROOT_DIR, '..', 'detections')

os.makedirs(OUTPUT_DIR, exist_ok=True)

model = YOLO(WEIGHTS_PATH)
print("Model class names:", model.names)

# DeepSORT
tracker = DeepSort(
    max_age=10,           
    n_init=5,               
    nms_max_overlap=1.0,   
    max_cosine_distance=0.20   
)


def draw_tracks(frame, tracks):
    for track in tracks:
        if not track.is_confirmed():
            continue
        x, y, w, h = track.to_ltwh()
        track_id = track.track_id
        cv2.rectangle(frame, (int(x), int(y)), (int(x+w), int(y+h)), (255, 0, 0), 2)
        cv2.putText(frame, f"ID {track_id}", (int(x), int(y)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)
    return frame


if __name__ == '__main__':
    cap = cv2.VideoCapture(VIDEO_PATH)
    frame_idx = 0
    print("Starting YOLO + DeepSORT tracking with player-only filter...")


    video_output_path = os.path.join(OUTPUT_DIR, 'test2.avi')
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_writer = None  

    while True:
        if frame_idx >= 500:
            break
        ret, frame = cap.read()
        if not ret:
            break

        
        results = model(frame)[0]
        detections = []

        for box in results.boxes:
            cls = int(box.cls)
            conf = float(box.conf)
            if model.names[cls] != 'player':
                continue
            if conf < 0.5:
                continue

            x1, y1, x2, y2 = box.xyxy[0].tolist()
            w = x2 - x1
            h = y2 - y1
            detections.append(([x1, y1, w, h], conf, 'player'))

        
        tracks = tracker.update_tracks(detections, frame=frame)

        if frame_idx < 500:
            out = draw_tracks(frame.copy(), tracks)
            print(frame_idx)

        
            # cv2.imwrite(os.path.join(OUTPUT_DIR, f"lowconflowage_{frame_idx:03d}.jpg"), out)

            
            if video_writer is None:
                height, width, _ = out.shape
                video_writer = cv2.VideoWriter(video_output_path, fourcc, 20.0, (width, height))

            
            video_writer.write(out)

        frame_idx += 1

    cap.release()
    if video_writer:
        video_writer.release()

    print(f"Saved {min(frame_idx, 200)} tracking frames and video to {OUTPUT_DIR}")
