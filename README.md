# Computer-Vision-Driven-Framework-for-Quantifying-Heat-Transfer-in-Dropwise-Condensation

This repository demonstrates how to train a YOLOv8 model using a custom dataset from Roboflow and then use the trained model to create an annotated tracking video. The script handles bounding box visualization, real-time preview, and video compilation from a sequence of images.

---
## **Droplet Detection**
## **Features**
- **Custom Model Training**: Train a YOLOv8 model on your custom dataset using `data.yaml` from Roboflow.
- **Inference on Test Images**: Perform inference on test images and save predicted results.
- **GPU Acceleration**: Supports CUDA-enabled devices for faster computation.

---

## **Requirements**
To run the code, ensure the following libraries and tools are installed:
- Python 3.8+
- PyTorch (with GPU support)
- Ultralytics YOLO (`pip install ultralytics`)
- OpenCV (`pip install opencv-python`)
- Matplotlib (`pip install matplotlib`)

---

## **Setup**
1. Clone the repository:
   ```bash
   git clone <repository_url>
   cd <repository_name>
   ```
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Ensure your dataset is organized as per YOLO's requirements and referenced in the `data.yaml` file.

---

## **Usage**

### 1. **Train the Model**
Modify the training parameters in the script as needed:
- `epochs`: Number of training iterations.
- `imgsz`: Image size for training.
- `batch`: Batch size for optimization.
- `lr0`: Initial learning rate.

Run the following code to train the model:
```python
model.train(
    data='path/to/data.yaml',
    epochs=50,
    imgsz=416,
    batch=2,
    name='yolov8_custom',
    device='0',  # Use GPU
    lr0=0.001
)
```

### 2. **Inference**
Load the trained weights and perform inference on test images:
```python
model = YOLO('path/to/weights/best.pt')
results = model('path/to/test/images')
```

### 3. **Save Predictions**
Predicted images are saved to a directory:
```python
save_dir = './predicted_images'
os.makedirs(save_dir, exist_ok=True)

for i, result in enumerate(results):
    save_path = os.path.join(save_dir, f"result_{i}.jpg")
    result.plot(save=True, filename=save_path)
```

---

## **Project Directory Structure**
Ensure the following structure for proper functionality:
```
project_root/
│
├── data/
│   ├── train/         # Training images
│   ├── val/           # Validation images
│   ├── test/          # Test images
│   └── data.yaml      # Dataset configuration file
│
├── runs/              # YOLO training runs and weights
│
├── predicted_images/  # Directory to save predictions
│
├── yolo.py            # Script for training and inference
└── README.md          # Project documentation
```

---

## **Notes**
- Ensure the `data.yaml` file is correctly configured with your dataset paths and class labels.
- Adjust training parameters (`epochs`, `batch size`, etc.) based on your hardware and dataset size.
- GPU acceleration requires CUDA-compatible hardware and a proper PyTorch installation.

---
## **Droplet Tracking Video Creation**
---

## **Features**
- **Bounding Box Visualization**: Draws YOLO-predicted bounding boxes on images.
- **Customizable Frame Rate**: Specify the desired FPS for the output video.
- **Video Compilation**: Combines annotated images into a seamless tracking video.
- **Real-Time Display**: Option to view the video while processing.

---

## **Usage**

### 1. **Input Data**
- **Image Folder**: Specify the folder containing images for video creation:
  ```python
  predicted_images_folder = 'path/to/image/folder'
  ```
- **Trained YOLO Model**: Specify the path to your trained YOLOv8 model:
  ```python
  trained_model_path = 'path/to/yolov8/weights/best.pt'
  ```

### 2. **Parameters**
- **Output Video Path**: Define the path for the output video:
  ```python
  output_video_file = './output_video.mp4'
  ```
- **Frame Rate**: Adjust the frames per second (FPS) for the video:
  ```python
  fps = 12  # Change this value as needed
  ```

### 3. **Run the Script**
Execute the script to generate the tracking video:
```bash
python video.py
```

### 4. **Output**
The script produces a video (`output_video.mp4`) with bounding boxes drawn on the detected droplets. During processing, the annotated frames are displayed in a real-time preview window.

---

## **How It Works**
1. **Image Loading**: The script reads all `.jpg` images from the specified folder.
2. **YOLO Detection**: For each image, the YOLO model predicts bounding boxes, classes, and confidence scores.
3. **Annotation**: The script overlays bounding boxes and labels on each frame.
4. **Video Compilation**: Frames are written sequentially to an output video file using OpenCV.


---
## **Droplet Tracking and Heat Transfer Calculation**
---

## **Features**
- **YOLOv8 Detection**: Detects droplets in image frames with a pre-trained YOLOv8 model.
- **Droplet Tracking**: Tracks droplets between consecutive frames using centroid proximity.
- **Heat Transfer Calculation**: Estimates heat transfer based on the droplet's growth rate and radius.
- **CSV Output**: Exports the results, including droplet properties and heat transfer, to a CSV file.

---

## **Setup**
1. Place your frames (images) in a directory, e.g., `E:/CS 499/Dataset/tracking_dataset/`.
2. Ensure you have a trained YOLOv8 model and update the model path in the script.

---

## **Usage**

### 1. **Input Data**
- **Frames Directory**: Update `frames_dir_path` in the script with the path to your image sequence.
- **Trained Model Path**: Update the YOLO model path in the script:
  ```python
  model = YOLO('path/to/weights/best.pt')
  ```

### 2. **Parameters**
You can adjust the following parameters for better performance:
- `pixels_per_mm`: Conversion ratio for pixels to millimeters.
- `min_radius_threshold`: Minimum droplet radius (in meters) to consider.
- `distance_threshold_mm`: Maximum distance (in mm) for tracking droplets between frames.

### 3. **Run the Script**
Execute the script to process the image sequence:
```bash
python main_calculations.py
```

### 4. **Output**
The script generates a CSV file, `droplet_radii_with_heat_transfer.csv`, with the following columns:
- **Frame**: Frame number.
- **Droplet_ID**: Unique identifier for each droplet.
- **Radius_pixels**: Droplet radius in pixels.
- **Radius_mm**: Droplet radius in millimeters.
- **Growth_rate_mm/s**: Droplet growth rate (mm/s).
- **Heat_transfer_J/s**: Heat transfer rate (Joules per second).

---

## **Algorithm Details**
1. **Detection**: Droplets are detected using YOLOv8, and bounding box dimensions are used to estimate the radius.
2. **Tracking**: Droplets in consecutive frames are matched based on their centroids using a KDTree and a distance threshold.
3. **Heat Transfer Calculation**:
   - Surface area is estimated using the radius.
   - Heat transfer is calculated using the formula:
     \[
     Q = L \cdot A \cdot \dot{r}
     \]
     where \( L \) is the latent heat of condensation, \( A \) is the surface area, and \( \dot{r} \) is the growth rate.

---

## **Acknowledgments**
- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) for the YOLO library.
- [RoboFlow](https://roboflow.com/) for dataset preparation and augmentation tools.

---

## **License**
This project is licensed under the MIT License. See the LICENSE file for details.

--- 

Happy detecting! 🚀
