Dimas Alvianto (23422009)

# Step 1: Install necessary libraries
!pip install ultralytics  # Install YOLOv8
!pip install matplotlib opencv-python-headless
!pip install roboflow

# Step 2: Import libraries
import matplotlib.pyplot as plt
from ultralytics import YOLO
from google.colab import files
import cv2
import numpy as np

!pip install roboflow

from roboflow import Roboflow
rf = Roboflow(api_key="p9AE4cfyWZVtKr7MenmF")
project = rf.workspace("dimas-alvianto-knkfe").project("plate-model-detection-2")
version = project.version(1)
dataset = version.download("yolov8")

import os

# Lihat folder tempat dataset diunduh
dataset_location = dataset.location  # dari RoboFlow download
print("Dataset downloaded to:", dataset_location)

from ultralytics import YOLO

# Buat model YOLOv8 baru
model = YOLO("yolov8n.pt")  # "yolov8n.pt" adalah versi YOLOv8 Nano

# Jalankan pelatihan dengan dataset
model.train(data="/content/plate-model-detection-2-1/data.yaml", epochs=3, imgsz=640)

# Ambil elemen pertama dari hasil prediksi
image_result = result[0]

# Menampilkan hasil deteksi
from IPython.display import Image, display
image_path_with_predictions = image_result.plot()  # Mengembalikan array gambar

# Tampilkan menggunakan Matplotlib
import matplotlib.pyplot as plt
plt.imshow(image_path_with_predictions)
plt.axis("off")
plt.show()

# Step 3: Upload image
uploaded = files.upload()
image_path = list(uploaded.keys())[0]

# Step 4: Load YOLO model
model = YOLO('yolov8n.pt')  # Use a pre-trained YOLOv8 model (nano version for speed)

# Step 5: Perform object detection
results = model(image_path)  # Run inference on the uploaded image

# Step 6: Visualize results
# Save the annotated image
annotated_img = results[0].plot()  # Create an annotated image (numpy array)

# Display the annotated image using matplotlib
plt.figure(figsize=(10, 10))
plt.imshow(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.title("YOLO Detected Objects")
plt.show()


