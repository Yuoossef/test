from ultralytics import YOLO
import streamlit as st
import os
from PIL import Image, ImageDraw, ImageFont
import tempfile
import requests

# Constants
FONT_URL = "https://github.com/google/fonts/blob/main/apache/roboto/Roboto-Regular.ttf?raw=true"
FONT_SIZE = 25

# Function to download a file from a URL
def download_file(url, save_path):
    response = requests.get(url)
    if response.status_code == 200:
        with open(save_path, 'wb') as file:
            file.write(response.content)
        print(f"File downloaded and saved to {save_path}")
    else:
        print(f"Failed to download file from {url}")

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

    # Download the font file
    font_path = "Roboto-Regular.ttf"
    if not os.path.exists(font_path):
        download_file(FONT_URL, font_path)

    font = ImageFont.truetype(font_path, FONT_SIZE)

    # Model URLs
    model_urls = [
        "https://github.com/Yuoossef/test/raw/main/Landmark%20Object%20detection.pt",
        "https://github.com/Yuoossef/test/raw/main/Keywords.pt",
        "https://github.com/Yuoossef/test/raw/main/Hieroglyph%20Net.pt",
        "https://github.com/Yuoossef/test/raw/main/Egypt%20Attractions.pt"
    ]

    # Download the models
    model_paths = []
    for url in model_urls:
        model_name = url.split("/")[-1]
        if not os.path.exists(model_name):
            download_file(url, model_name)
        model_paths.append(model_name)

    # Load YOLO models
    hieroglyph_model = load_yolov11_model(model_paths[1])
    attractions_model = load_yolov11_model(model_paths[3])
    landmarks_model = load_yolov11_model(model_paths[0])
    hieroglyph_net_model = load_yolov11_model(model_paths[2])

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
