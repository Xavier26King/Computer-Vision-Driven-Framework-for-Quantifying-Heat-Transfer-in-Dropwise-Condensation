from ultralytics import YOLO
import cv2
import os
import glob

def draw_bounding_boxes(frame, detections):
    """
    Draw bounding boxes on the frame.
    :param frame: Image frame
    :param detections: YOLO detections
    """
    for detection in detections:
        x1, y1, x2, y2 = map(int, detection.xyxy[0])  # Extract bounding box coordinates
        confidence = detection.conf[0]  # Confidence score
        label = f"{detection.cls[0]} {confidence:.2f}"  # Label with class and confidence
        
        # Draw the rectangle
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # Add the label
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return frame

def create_tracking_video_with_model(image_folder, model_path, output_video_path, fps=12):
    # Load the YOLO model
    model = YOLO(model_path)
    
    # Get all image paths from the folder and sort them
    image_paths = sorted(glob.glob(os.path.join(image_folder, "*.jpg")))

    # Check if the folder contains any images
    if not image_paths:
        print("No images found in the specified folder.")
        return

    # Read the first image to get the frame dimensions
    first_frame = cv2.imread(image_paths[0])
    height, width, layers = first_frame.shape

    # Initialize the video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4 files
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    for image_path in image_paths:
        # Read each frame
        frame = cv2.imread(image_path)

        # Perform YOLO prediction
        results = model(frame)

        # Draw bounding boxes on the frame
        for result in results:
            frame = draw_bounding_boxes(frame, result.boxes)

        # Write the frame to the video
        video_writer.write(frame)

        # Display the frame
        cv2.imshow("Droplet Tracking Video", frame)
        if cv2.waitKey(30) & 0xFF == ord('q'):  # Press 'q' to exit early
            break

    # Release the video writer and close OpenCV windows
    video_writer.release()
    cv2.destroyAllWindows()

# Define paths
predicted_images_folder = 'E:/CS 499/Dataset/set_3/set_3_bare_copper_gleco'  # Folder with predicted images
trained_model_path = 'E:/CS 499/Real Project/runs/detect/yolov8_custom2/weights/best.pt'  # Path to trained YOLO model
output_video_file = './droplet_tracking_video.mp4'  # Output video file path

# Create the tracking video with bounding boxes
create_tracking_video_with_model(predicted_images_folder, trained_model_path, output_video_file)
