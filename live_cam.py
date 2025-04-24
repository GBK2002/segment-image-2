import os
import cv2
import torch
import numpy as np
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
import time
import threading
from queue import Queue
from torch.cuda.amp import autocast

# Define paths
sam2_checkpoint = "C:\\Users\\khand\\IC\\segment-anything-2\\checkpoints\\sam2.1_hiera_tiny.pt"
model_cfg = "C:\\Users\\khand\\IC\\segment-anything-2\\sam2_configs\\sam2_hiera_l.yaml"
output_video = "tracked_object_video.avi"  # Path for the output video

# Load SAM2 model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)
predictor = SAM2ImagePredictor(sam2_model)

# Global variables
selected_point = None
recording = False
frames_queue = Queue()  # Queue to store frames for processing
masked_frames = []  # List to store processed frames
fps = 15  # Reduced FPS for faster processing

# Mouse callback function to capture the selected point
def select_point(event, x, y, flags, param):
    global selected_point, recording
    if event == cv2.EVENT_LBUTTONDOWN:
        selected_point = [x, y]
        recording = True  # Start recording once an object is selected
        print(f"Selected point: ({x}, {y})")

# Record video in a separate thread and put frames into a queue
def record_video(cap, duration=10):
    total_frames = duration * fps
    frame_count = 0

    while frame_count < total_frames:
        ret, frame = cap.read()
        if not ret:
            break

        # Keep resolution at 640x480 for tracking accuracy
        frame = cv2.resize(frame, (640, 480))
        frames_queue.put(frame)  # Add frame to the queue
        frame_count += 1

    print(f"Recorded {frame_count} frames.")
    frames_queue.put(None)  # Signal end of recording

# Function to apply mask on frames using SAM2 model
def mask_frames():
    first_frame = True
    last_mask = None

    # Use CUDA streams to overlap data transfer and computation
    stream = torch.cuda.Stream()
    torch.cuda.synchronize()  # Ensure previous GPU operations are complete

    while True:
        frame = frames_queue.get()  # Get the next frame from the queue
        if frame is None:
            break  # End of frames

        # Perform image processing with torch.no_grad for faster inference
        with torch.no_grad(), torch.cuda.stream(stream), autocast():
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            if first_frame:
                # Initialize tracking with the selected point
                predictor.set_image(frame_rgb)
                input_point = np.array([selected_point])
                input_label = np.array([1])  # Positive point
                masks, _, _ = predictor.predict(
                    point_coords=input_point,
                    point_labels=input_label,
                    multimask_output=False
                )
                first_frame = False
            else:
                # Calculate a bounding box from the last mask
                if last_mask is not None:
                    x, y, w, h = cv2.boundingRect(last_mask.astype(np.uint8))
                    box = np.array([x, y, x + w, y + h], dtype=np.float32)
                else:
                    print("No previous mask found, skipping frame.")
                    continue

                # Set the new frame and predict using the bounding box
                predictor.set_image(frame_rgb)
                masks, _, _ = predictor.predict(
                    box=box,
                    multimask_output=False
                )

            if masks is not None and len(masks) > 0:
                mask = masks[0]
                mask = mask.astype(np.uint8) * 255  # Convert to 0-255 range

                # Update the last mask
                last_mask = mask

                # Create a colored mask and overlay
                colored_mask = np.zeros_like(frame, dtype=np.uint8)
                colored_mask[mask > 0] = [0, 255, 0]  # Green mask
                overlayed_frame = cv2.addWeighted(frame, 0.6, colored_mask, 0.4, 0)
                masked_frames.append(overlayed_frame)  # Store the processed frame
            else:
                print("No mask found for the frame")

    torch.cuda.synchronize()  # Ensure all GPU tasks are completed

# Convert processed frames to a video
def frames_to_video(fps=15):
    if not masked_frames:
        print("No frames found to stitch into a video.")
        return

    # Get frame dimensions
    height, width, _ = masked_frames[0].shape

    # Define video writer
    video_writer = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc(*'XVID'), fps, (width, height))

    # Add frames to the video
    for frame in masked_frames:
        video_writer.write(frame)

    video_writer.release()
    print(f"Video saved to {output_video}")

# Main workflow
def main():
    start_time = time.time()
    global recording
    cap = cv2.VideoCapture(0)  # Start webcam feed

    # Set the mouse callback function to select a point in the live feed
    cv2.namedWindow('Live Feed')
    cv2.setMouseCallback('Live Feed', select_point)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Show the live feed until a point is selected
        cv2.imshow('Live Feed', frame)

        # If recording has started, break the live feed loop
        if recording:
            break

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            return

    # Start recording and processing in separate threads
    record_thread = threading.Thread(target=record_video, args=(cap, 15))
    mask_thread = threading.Thread(target=mask_frames)

    record_thread.start()
    mask_thread.start()

    record_thread.join()  # Wait for recording to finish
    mask_thread.join()    # Wait for processing to finish

    # Release the webcam
    cap.release()
    cv2.destroyAllWindows()

    # Convert frames to video
    frames_to_video(fps=fps)

    end_time = time.time()
    print(f"Total runtime: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    main()
