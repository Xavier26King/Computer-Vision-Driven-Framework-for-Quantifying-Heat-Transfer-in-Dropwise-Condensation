from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
import os
import torch

# if __name__ == '__main__':
#     model = YOLO('yolov8s.pt')
#     print(torch.cuda.is_available())
#     results = model.train(
#         data='E:/CS 499/Real Project/droplet-detection-4/data.yaml',  # Path to the data.yaml file from RoboFlow
#         epochs=50,  # Number of training epochs
#         imgsz=416,  # Image size (default 640, can be changed)
#         batch=2,  # Batch size (adjust based on your hardware)
#         name='yolov8_custom',  # Name for saving the run results
#         val=False,  # Perform validation during training
#         device='0',  # Use GPU if available
#         lr0=0.001
#     )

# After training, load the best weights and test on a new image
model = YOLO('E:/CS 499/Real Project/runs/detect/yolov8_custom2/weights/best.pt')  # Load the trained model

# Perform inference on a test image
results = model('E:/CS 499/Real Project/test')  # Replace with the path to your test image

# save the results in form of image in one folder
save_dir = './predicted_images'
os.makedirs(save_dir, exist_ok=True)  # Create directory if it doesn't exist

# Save the results in the specified directory
for i, result in enumerate(results):
    # Generate the file path for saving each result
    save_path = os.path.join(save_dir, f"result_{i}.jpg")
    result.plot(save=True, filename=save_path)
