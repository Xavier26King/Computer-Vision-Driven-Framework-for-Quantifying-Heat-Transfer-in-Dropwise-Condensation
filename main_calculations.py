import cv2
import os
import glob
import csv
from ultralytics import YOLO
import torch
import numpy as np
from scipy.spatial import KDTree

# # Function to measure droplet properties from the image
# def measure_droplet_properties(image, pixels_per_mm):
#     gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     _, droplet_mask = cv2.threshold(gray_image, 45, 255, cv2.THRESH_BINARY)
#     contours, _ = cv2.findContours(droplet_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#     droplet_data = []

#     for contour in contours:
#         x, y, w, h = cv2.boundingRect(contour)
#         droplet_radius_pixels = min(w, h) / 2.0  # approximating radius from bounding rectangle
#         droplet_radius_mm = droplet_radius_pixels / pixels_per_mm  # convert to mm

#         if droplet_radius_mm >= min_radius_threshold:  
#             droplet_data.append({
#                 'radius': droplet_radius_pixels,  # in pixels
#                 'radius_mm': droplet_radius_mm,  # in mm
#                 'centroid': (x + w // 2, y + h // 2)
#             })

#     return droplet_data

# Constants for heat transfer calculation
latent_heat_of_condensation_water = 2.26e6  # J/kg
time_increment = 1 / 10.0  # Assuming 30 FPS (time between frames in seconds)

# Function to calculate heat transfer
def calculate_heat_transfer(droplet_radius_m, growth_rate):
    surface_area = 4 * np.pi * (droplet_radius_m ** 2)
    heat_transfer = latent_heat_of_condensation_water * surface_area * growth_rate * 1000  # in J/s
    return heat_transfer

# Function to track droplets between frames with additional metrics
def track_droplets(prev_droplets, curr_droplets, pixels_per_mm):
    global next_droplet_id
    global distance_threshold_mm

    distance_threshold_pixels = distance_threshold_mm * pixels_per_mm

    if not prev_droplets:
        for curr_droplet in curr_droplets:
            curr_droplet['growth_rate'] = 0
            curr_droplet['heat_transfer'] = calculate_heat_transfer(curr_droplet['radius_mm'] / 1000, 0)
            curr_droplet['id'] = next_droplet_id
            next_droplet_id += 1
        return curr_droplets

    prev_centroids = [droplet['centroid'] for droplet in prev_droplets]
    curr_centroids = [droplet['centroid'] for droplet in curr_droplets]

    if not prev_centroids or not curr_centroids:
        return curr_droplets

    tree = KDTree(prev_centroids)  # For nearest neighbor search
    distances, indices = tree.query(curr_centroids)

    used_ids = set()

    for i, curr_droplet in enumerate(curr_droplets):
        prev_droplet = prev_droplets[indices[i]]
        distance = distances[i]

        if distance < distance_threshold_pixels:
            # Calculate growth rate and heat transfer
            growth_rate = (curr_droplet['radius_mm'] - prev_droplet['radius_mm']) / time_increment  # mm/s
            if growth_rate < -1e-6:  # Ignore slight negative growth rates due to noise
                growth_rate = 0

            curr_droplet['growth_rate'] = growth_rate
            curr_droplet['heat_transfer'] = calculate_heat_transfer(curr_droplet['radius_mm'] / 1000, growth_rate)

            # Assign existing or new droplet ID
            if prev_droplet['id'] not in used_ids:
                curr_droplet['id'] = prev_droplet['id']
                used_ids.add(prev_droplet['id'])
            else:
                curr_droplet['id'] = next_droplet_id
                next_droplet_id += 1
        else:
            # Assign new ID and set growth rate/heat transfer to zero
            curr_droplet['id'] = next_droplet_id
            curr_droplet['growth_rate'] = 0
            curr_droplet['heat_transfer'] = calculate_heat_transfer(curr_droplet['radius_mm'] / 1000, 0)
            next_droplet_id += 1

    return curr_droplets

# Function to measure droplet properties using YOLO model predictions
def measure_droplet_properties_with_yolo(frame, model, pixels_per_mm, min_radius_threshold):
    results = model(frame)
    predictions = results[0].boxes.xywh  # Extract bounding box predictions (center_x, center_y, width, height)
    scores = results[0].boxes.conf  # Confidence scores
    classes = results[0].boxes.cls  # Class predictions

    droplet_data = []

    for i, box in enumerate(predictions):
        if classes[i] == 0 and scores[i] > 0.5:  # Assuming class '0' corresponds to droplets
            x_center, y_center, w, h = box
            droplet_radius_pixels = min(w, h) / 2.0  # approximating radius
            droplet_radius_mm = droplet_radius_pixels / pixels_per_mm  # convert to mm

            if droplet_radius_mm >= min_radius_threshold:
                droplet_data.append({
                    'radius': droplet_radius_pixels,  # in pixels
                    'radius_mm': droplet_radius_mm,  # in mm
                    'centroid': (int(x_center), int(y_center))
                })

    return droplet_data

# Main processing loop with YOLO detection
frames_dir_path = "E:/CS 499/Dataset/tracking_dataset/"
output_csv_path = "droplet_radii_with_heat_transfer.csv"
frame_files = sorted(glob.glob(os.path.join(frames_dir_path, "*.jpg")))

if not frame_files:
    print("No frames found in the directory.")
else:
    print(f"Found {len(frame_files)} frames.")

    model = YOLO('E:/CS 499/Real Project/runs/detect/yolov8_custom2/weights/best.pt')
    print(torch.cuda.is_available())

    prev_droplets = []

    pixels_per_mm = 10
    min_radius_threshold = 0.005  # Minimum radius in meters
    next_droplet_id = 0
    distance_threshold_mm = 2

    with open(output_csv_path, mode='w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['Frame', 'Droplet_ID', 'Radius_pixels', 'Radius_mm', 'Growth_rate_mm/s', 'Heat_transfer_J/s'])

        for frame_idx, frame_file in enumerate(frame_files):
            print(f"Processing frame {frame_idx + 1}/{len(frame_files)}: {frame_file}")
            frame = cv2.imread(frame_file)

            if frame is None:
                print(f"Error loading frame {frame_file}")
                continue

            # Measure droplet properties using YOLO model
            droplet_data = measure_droplet_properties_with_yolo(frame, model, pixels_per_mm, min_radius_threshold)
            droplet_data = track_droplets(prev_droplets, droplet_data, pixels_per_mm)

            for droplet in droplet_data:
                csv_writer.writerow([
                    frame_idx + 1, int(droplet['id']), float(droplet['radius']), float(droplet['radius_mm']),
                    float(droplet['growth_rate']), float(droplet['heat_transfer'])
                ])

            prev_droplets = droplet_data

        print(f"Droplet data saved to {output_csv_path}")

    cv2.destroyAllWindows()
