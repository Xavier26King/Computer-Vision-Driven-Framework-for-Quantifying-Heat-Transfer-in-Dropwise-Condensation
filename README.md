# Computer-Vision-Driven-Framework-for-Quantifying-Heat-Transfer-in-Dropwise-Condensation

# README: Droplet Detection Using YOLOv8

This repository provides code and resources for training and testing a custom object detection model using **YOLOv8**. The project focuses on detecting droplets in images, leveraging the capabilities of the **Ultralytics YOLO library** and GPU acceleration for efficient training and inference.

---

## **Features**
- **Custom Model Training**: Train a YOLOv8 model on your dataset using `data.yaml` for custom object detection tasks.
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
├── main.py            # Script for training and inference
└── README.md          # Project documentation
```

---

## **Notes**
- Ensure the `data.yaml` file is correctly configured with your dataset paths and class labels.
- Adjust training parameters (`epochs`, `batch size`, etc.) based on your hardware and dataset size.
- GPU acceleration requires CUDA-compatible hardware and a proper PyTorch installation.

---

## **Acknowledgments**
- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) for the YOLO library.
- [RoboFlow](https://roboflow.com/) for dataset preparation and augmentation tools.

---

## **License**
This project is licensed under the MIT License. See the LICENSE file for details.

--- 

Happy detecting! 🚀
