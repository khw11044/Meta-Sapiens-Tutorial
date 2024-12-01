import cv2
import numpy as np
from tqdm import tqdm
from PIL import Image
import torch
from utils.utils import *  # Placeholder for necessary utility functions

def process_webcam(pose_model, show=True):
    """
    Process webcam input for 2D Human Pose Estimation.

    Args:
        pose_model: The pose estimation model (TorchScript or equivalent).
        show (bool): Whether to display the video frames in real-time.
    """
    # Open the webcam
    cap = cv2.VideoCapture(0)  # 0은 기본 웹캠
    if not cap.isOpened():
        print("Error: Could not access the webcam.")
        return

    print("Processing webcam input... Press 'q' to exit.")

    # Process each frame
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to read frame from webcam.")
            break
        

        # Convert OpenCV frame (BGR) to PIL Image (RGB)
        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # Perform pose estimation
        pose_image = get_pose(pil_image, pose_model)  # Ensure `get_pose` is defined

        # Convert PIL Image (RGB) back to OpenCV format (BGR)
        pose_frame = cv2.cvtColor(np.array(pose_image), cv2.COLOR_RGB2BGR)

        # Show the frame
        if show:
            cv2.imshow('Pose Estimation', pose_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit
                break

    # Clean up
    cap.release()
    if show:
        cv2.destroyAllWindows()

    print("Webcam processing ended.")

# Example usage
if __name__ == "__main__":
    # Load your pose estimation model here
    TASK = 'pose'
    VERSION = 'sapiens_0.3b'
    model_path = get_model_path(TASK, VERSION)  # Ensure `get_model_path` is defined
    pose_model = torch.jit.load(model_path)
    pose_model.eval()
    pose_model.to("cuda")

    process_webcam(pose_model, show=True)
