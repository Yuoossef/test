from ultralytics import YOLO
import streamlit as st
import os
from PIL import Image, ImageDraw, ImageFont
import tempfile
import requests

# Constants
FONT_PATH = "C:\\Windows\\Fonts\\arial.ttf"
FONT_SIZE = 25

# Function to download models from GitHub
def download_model(url, save_path):
    response = requests.get(url)
    if response.status_code == 200:
        with open(save_path, 'wb') as file:
            file.write(response.content)
        print(f"Model downloaded and saved to {save_path}")
    else:
        print(f"Failed to download model from {url}")

# Function to load YOLOv11 models
def load_yolov11_model(model_path):
    try:
        model = YOLO(model_path)
        print(f"Model loaded successfully from {model_path}")
        return model
    except Exception as e:
        print(f"Error loading YOLO model from {model_path}: {e}")
        return None

# Function to run inference and draw bounding boxes
def run_inference_unique(model, image, draw, font, processed_classes):
    try:
        results = model.predict(source=image, save=False)
        detected_classes = []

        for result in results:
            if hasattr(result, 'boxes') and result.boxes is not None:
                for box in result.boxes:
                    if hasattr(box, 'cls') and hasattr(box, 'xyxy'):
                        cls_id = int(box.cls)
                        cls_name = model.names[cls_id]

                        # Check if class already processed
                        if cls_name not in processed_classes:
                            detected_classes.append(cls_name)
                            processed_classes.add(cls_name)  # Mark class as processed

                            # Draw the bounding box
                            x1, y1, x2, y2 = box.xyxy.tolist()[0]
                            draw.rectangle([x1, y1, x2, y2], outline="blue", width=2)
                            draw.text((x1, y1 - 10), cls_name, fill="white", font=font)

        return detected_classes
    except Exception as e:
        print(f"Error during inference: {e}")
        return []

# Streamlit interface
st.title("YOLOv11 Model Testing")
uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    st.image(uploaded_file, caption="Uploaded Image")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
        temp_file.write(uploaded_file.read())
        image_path = temp_file.name

    original_image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(original_image)
    font = ImageFont.truetype(FONT_PATH, FONT_SIZE)

    # Model URLs
    model_urls = [
        "https://github.com/Yuoossef/test/raw/main/Landmark%20Object%20detection.pt",
        "https://github.com/Yuoossef/test/raw/main/Keywords.pt",
        "https://github.com/Yuoossef/test/raw/main/Hieroglyph%20Net.pt",
        "https://github.com/Yuoossef/test/raw/main/Egypt%20Attractions.pt"
    ]
    
    # Local paths to save the models
    save_paths = [
        "C:\\path\\to\\your\\model\\Landmark Object detection.pt",
        "C:\\path\\to\\your\\model\\Keywords.pt",
        "C:\\path\\to\\your\\model\\Hieroglyph Net.pt",
        "C:\\path\\to\\your\\model\\Egypt Attractions.pt"
    ]

    # Download the models
    for url, path in zip(model_urls, save_paths):
        download_model(url, path)

    # Load YOLO models
    hieroglyph_model = load_yolov11_model(save_paths[1])
    attractions_model = load_yolov11_model(save_paths[3])
    landmarks_model = load_yolov11_model(save_paths[0])
    hieroglyph_net_model = load_yolov11_model(save_paths[2])

    # Tracking processed classes
    processed_classes = set()
    all_results = []

    # Process each model
    if hieroglyph_model:
        all_results.extend(run_inference_unique(hieroglyph_model, image_path, draw, font, processed_classes))
    if attractions_model:
        all_results.extend(run_inference_unique(attractions_model, image_path, draw, font, processed_classes))
    if landmarks_model:
        all_results.extend(run_inference_unique(landmarks_model, image_path, draw, font, processed_classes))
    if hieroglyph_net_model:
        all_results.extend(run_inference_unique(hieroglyph_net_model, image_path, draw, font, processed_classes))

    # Extract unique classes
    unique_results = list(set(all_results))
    st.write("All Detected Unique Classes:", unique_results)

    # Save and display the final annotated image
    final_annotated_path = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg").name
    original_image.save(final_annotated_path)
    st.image(final_annotated_path, caption="Final Annotated Image")
    print(unique_results)
