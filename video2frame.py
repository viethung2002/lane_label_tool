import cv2
import os
import argparse

def extract_frames(video_path, output_folder):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    # Check if video opened successfully
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return

    # Get the frames per second (fps) of the video
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:  # Avoid division by zero
        print("Error: Could not determine FPS.")
        return
    
    frame_interval = int(fps)  # Capture one frame per second

    frame_count = 0
    saved_frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            frame_filename = os.path.join(output_folder, f"{saved_frame_count:04d}.jpg")
            cv2.imwrite(frame_filename, frame)
            saved_frame_count += 1

        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()
    print(f"Frames extracted and saved to {output_folder}")

# Argument parser setup
parser = argparse.ArgumentParser(description="Extract frames from a video")
parser.add_argument('--video_path', type=str, required=True, help="Path to the video file")
parser.add_argument('--output_frames', type=str, required=True, help="Path to save extracted frames")

# Parse the arguments
args = parser.parse_args()

# Run the function with the provided arguments
extract_frames(args.video_path, args.output_frames)
