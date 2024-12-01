import cv2
import numpy as np
from tqdm import tqdm
from PIL import Image
import torch
from utils.utils import *

# Placeholder for `get_pose` function and other necessary imports from your project.
# Ensure `get_pose` and any other dependencies are available in the same script or imported properly.

def process_video(input_path, output_path, pose_model, show=False):
    """
    Process a video for 2D Human Pose Estimation and save the output.

    Args:
        input_path (str): Path to the input video file.
        output_path (str): Path to save the processed video file.
        pose_model: The pose estimation model (TorchScript or equivalent).
        show (bool): Whether to display the video frames in real-time.
    """
    # Open the input video
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {input_path}")
        return

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Initialize the video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'mp4v' codec
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Process each frame
    print("Processing video...")
    for _ in tqdm(range(frame_count), desc="Frames processed"):
        ret, frame = cap.read()
        if not ret:
            break
        # frame =  cv2.resize(frame, (360, 640))
        # Convert OpenCV frame (BGR) to PIL Image (RGB)
        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # Perform pose estimation
        pose_image = get_pose(pil_image, pose_model)  # Ensure `get_pose` is defined

        # Convert PIL Image (RGB) back to OpenCV format (BGR)
        pose_frame = cv2.cvtColor(np.array(pose_image), cv2.COLOR_RGB2BGR)

        # Write the frame to the output video
        out.write(pose_frame)

        # Optionally show the frame
        if show:
            cv2.imshow('Pose Estimation', pose_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit early
                break

    # Clean up 
    cap.release()
    out.release()
    if show:
        cv2.destroyAllWindows()

    print(f"Output video saved to {output_path}")

# Example usage
if __name__ == "__main__":
    # input_video_path = "example1.mov"
    input_video_path = "./examples/example1.mov"
    output_video_path = "output.mp4"

    # Load your pose estimation model here
    TASK = 'pose'
    VERSION = 'sapiens_0.3b'
    model_path = get_model_path(TASK, VERSION)  # Ensure `get_model_path` is defined
    pose_model = torch.jit.load(model_path)
    pose_model.eval()
    pose_model.to("cuda")

    process_video(input_video_path, output_video_path, pose_model, show=True)
