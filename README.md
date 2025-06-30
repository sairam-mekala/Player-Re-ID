# Player Re-Identification (Single-Feed)

A Python pipeline that uses YOLOv11 for detection and DeepSORT for multi-object tracking to maintain consistent player IDs over a 15-second sports clip—even through occlusions.

---

## Repository Structure

```
player-reid/
├── data/                 # Input video: 15sec_input_720p.mp4
├── weights/              # YOLOv11 model weights (download below)
├── detections/           # Output images & demo video
├── src/                  # Source code
│   ├── detect.py         # Day 1: YOLOv11 detection
│   ├── track.py          # Day 2–3: YOLOv11 + DeepSORT tracking
│   └── utils.py          # (optional) helper functions
├── venv/                 # Python virtual environment
├── README.md             # this file
└── report.md             # methodology, challenges, results
```

---

## Download Assets

1. **YOLOv11 model weights**  
   Download the `yolov11.pt` file (a fine-tuned Ultralytics YOLOv11 for players & ball) from Google Drive:  
   https://drive.google.com/file/d/1-5fOSHOSB9UXyP_enOoZNAMScrePVcMD/view

2. **Input video**  
   Place `15sec_input_720p.mp4` in the `data/` folder.

---

## Setup and Dependencies

### 1. Clone this repository

```bash
git clone https://github.com/<your-username>/player-reid.git
cd player-reid
```

### 2. Create & activate a virtual environment

- **Windows**
```bash
python -m venv venv
venv\Scripts\activate
```

- **macOS/Linux**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install required Python packages

```bash
pip install --upgrade pip
pip install torch torchvision ultralytics opencv-python numpy deep_sort_realtime
```

### 4. DeepSORT used

This project uses [`deep_sort_realtime`](https://github.com/levan92/deep_sort_realtime) for real-time object re-identification and tracking.

---

## How to Run

### 1. Run detection only

```bash
cd src
python detect.py
```

- Runs YOLOv11 detection on the first 10 frames.
- Saves `det_000.jpg` to `det_009.jpg` in `detections/`.

### 2. Run full tracking with DeepSORT

```bash
python track.py
```

- Tracks players in the video.
- Outputs up to 200 frames and a video file (`avgage_lowoverlap_avgcos.avi`) into `detections/`.

---

## Output Example

After running `track.py`, check the `detections/` folder for:
- Tracked frames with bounding boxes and ID labels
- A demo video file combining all tracked frames

---

## Report

See [report.md](report.md) for:

- Approach & methodology
- Techniques tried and their outcomes
- Challenges and debugging process
- Limitations and future improvements
