# Report: Player Re-Identification (Single-Feed)

## 1. Approach and Methodology

**Objective:** Maintain consistent player IDs over a 15-second sports clip using YOLOv11 for detection and DeepSORT for tracking and re-identification.

1. **Detection**  
   - Loaded a fine-tuned YOLOv11 model (`yolov11.pt`) trained on players & ball.  
   - Ran inference per frame with a configurable confidence (`conf`) and NMS (`iou`) threshold.  
   - Filtered outputs by class name (`'player'`) and minimum bounding-box area.

2. **Tracking & Re-Identification**  
   - Integrated the `deep_sort_realtime` package for multi-object tracking.  
   - Converted each detection to the format `([x, y, w, h], confidence, 'player')`.  
   - Tuned DeepSORT parameters (`n_init`, `max_age`, `max_cosine_distance`, `nms_max_overlap`) to balance precision vs. recall.  
   - Added temporal smoothing: when a confirmed track briefly disappears or shrinks below area threshold, re-draw its last known box in a distinct color to maintain visual continuity.

3. **Output**  
   - Saved the first 200 tracked frames with bounding boxes and ID labels.  
   - Compiled those frames into a demo video (`.avi`) for easy review.

---

## 2. Techniques Tried and Outcomes

| Technique                                         | Outcome                                                                                 |
|---------------------------------------------------|-----------------------------------------------------------------------------------------|
| **Strict thresholds** (`conf ≥ 0.6`, `area ≥ 5000`)      | Fewer false positives (grass, ball), but many players dropped when overlapping or occluded. |
| **Looser thresholds** (`conf ≥ 0.4`, `area ≥ 2000`)      | Captured more players but introduced some spurious detections.                         |
| **Tighter appearance matching** (`max_cosine ≤ 0.15`)   | Reduced ID switches, but risked splitting tracks when appearance changed.              |
| **Temporal smoothing**                              | Improved visual continuity through occlusions; required careful tuning of buffer logic. |

---

## 3. Challenges Encountered

1. **Custom Class Indices**  
   - Original assumption that `cls==0` meant player was incorrect (it was “ball”). Needed to use `model.names` to filter by label rather than index.

2. **False Positives on Grass & Ball**  
   - Without area/​confidence filtering, small noise boxes cluttered the output. Addressed by adding minimum box area and raising confidence threshold.

3. **ID Switching on Overlap**  
   - When two players crossed or occluded each other, appearance embeddings changed, causing DeepSORT to reassign IDs. Solved partially with looser cosine distance and temporal smoothing.

4. **Balancing Precision vs. Recall**  
   - Stricter settings reduced junk detections but missed real players. Looser settings captured all players but needed smoothing to avoid jitter.

---

## 4. Remaining Work & Future Directions

- **Incomplete Aspects:**  
  - Current smoothing is heuristic (re-drawing last box); a learned re-ID embedding network could yield more robust association.  
  - Edge cases such as extremely fast motion and severe occlusion still cause ID switches.

- **With More Time/Resources:**  
  1. **Integrate a dedicated Re-ID model** (e.g., a small ResNet-based person-ReID network) to generate richer appearance embeddings.  
  2. **Camera motion compensation** or homography-based stabilization to reduce jitter.  
  3. **Quantitative evaluation** using manually labeled ground-truth to compute IDF1 and MOTA metrics.  
  4. **Extend to cross-camera scenarios** by adding geometric calibration and feature matching across views.

---

*End of Report*  
