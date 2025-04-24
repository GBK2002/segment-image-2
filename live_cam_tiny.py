import os
import cv2
import torch
import numpy as np
from sam2.build_sam import build_sam2  # Updated for SAM2.1
from sam2.sam2_image_predictor import SAM2ImagePredictor  # Updated Predictor for 2.1
import time
import threading
from queue import Queue
from torch.cuda.amp import autocast

# 🔹 Paths for Sierra-Tiny Model (Updated for SAM2.1)
BASE_DIR = "C:\\Users\\khand\\IC\\segment-anything-2"
sam2_checkpoint = os.path.join(BASE_DIR, "checkpoints", "sam2.1_hiera_tiny.pt")
model_cfg = os.path.join(BASE_DIR, "sam2", "sam2_hiera_t.yaml")  # ✅ Corrected Path
output_video = os.path.join(BASE_DIR, "tracked_object_video.avi")

# 🔹 Validate Config File Existence
if not os.path.exists(model_cfg):
    raise FileNotFoundError(f"Config file missing: {model_cfg}")

# 🔹 Load SAM2.1 Model (Force Ignore Unexpected Keys)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)

# ✅ Override Model Checkpoint Loading
state_dict = torch.load(sam2_checkpoint, map_location=device)

# 🔹 Manually Remove Unexpected Keys
expected_keys = sam2_model.state_dict().keys()
filtered_state_dict = {k: v for k, v in state_dict.items() if k in expected_keys}

# 🔹 Load the filtered state dict
missing_keys, unexpected_keys = sam2_model.load_state_dict(filtered_state_dict, strict=False)

# ✅ Print ignored keys
if unexpected_keys:
    print(f"⚠️ Warning: Ignored unexpected keys: {unexpected_keys}")

predictor = SAM2ImagePredictor(sam2_model)

# 🎯 Mouse callback to select an object
def select_point(event, x, y, flags, param):
    global selected_point, recording
    if event == cv2.EVENT_LBUTTONDOWN:
        selected_point = [x, y]
        recording = True
        print(f"Selected point: ({x}, {y})")

# 🎥 Record video frames into queue
def record_video(cap, duration=10):
    total_frames = duration * 10
    frame_count = 0

    while frame_count < total_frames:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (640, 480))
        frames_queue.put(frame)
        frame_count += 1

    print(f"Recorded {frame_count} frames.")
    frames_queue.put(None)

# 🖼️ Apply SAM2.1 segmentation
def mask_frames():
    first_frame = True
    last_mask = None
    torch.cuda.synchronize()

    while True:
        frame = frames_queue.get()
        if frame is None:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        with torch.no_grad(), autocast():
            if first_frame and selected_point is not None:
                predictor.set_image(frame_rgb)
                input_point = np.array([selected_point])
                input_label = np.array([1])

                masks, _, _ = predictor.predict(
                    point_coords=input_point,
                    point_labels=input_label,
                    multimask_output=False
                )
                first_frame = False
            else:
                if last_mask is not None:
                    x, y, w, h = cv2.boundingRect(last_mask.astype(np.uint8))
                    box = np.array([x, y, x + w, y + h], dtype=np.float32)
                else:
                    print("No previous mask found; skipping frame.")
                    continue

                predictor.set_image(frame_rgb)
                masks, _, _ = predictor.predict(
                    box=box,
                    multimask_output=False
                )

            if masks is not None and len(masks) > 0:
                mask = masks[0].astype(np.uint8) * 255
                last_mask = mask  # Update last mask

# 🏁 Main function
def main():
    cap = cv2.VideoCapture(0)
    cv2.namedWindow('Live Feed')
    cv2.setMouseCallback('Live Feed', select_point)

    record_thread = threading.Thread(target=record_video, args=(cap, 10))
    mask_thread = threading.Thread(target=mask_frames)

    record_thread.start()
    mask_thread.start()

    while not recording:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow('Live Feed', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    record_thread.join()
    mask_thread.join()

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
