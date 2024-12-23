from ultralytics import YOLO
import streamlit as st
import os
from PIL import Image, ImageDraw, ImageFont
import tempfile

FONT_PATH = "arial.ttf"
FONT_SIZE = 20

def load_yolov11_model(model_path):
    try:
        model = YOLO(model_path)
        print(f"Model loaded successfully from {model_path}")
        return model
    except Exception as e:
        print(f"Error loading YOLO model from {model_path}: {e}")
        return None


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


    hieroglyph_model = load_yolov11_model(r"C:\\Users\\youssef azam\\Downloads\\Keywords.pt")
    attractions_model = load_yolov11_model(r"C:\\Users\\youssef azam\\Downloads\\Egypt Attractions.pt")
    landmarks_model = load_yolov11_model(r"C:\\Users\\youssef azam\\Downloads\\Landmark Object detection.pt")
    hieroglyph_net_model = load_yolov11_model(r"C:\\Users\\youssef azam\\Downloads\\Hieroglyph Net.pt")

    processed_classes = set()
    all_results = []


    if hieroglyph_model:
        all_results.extend(run_inference_unique(hieroglyph_model, image_path, draw, font, processed_classes))
    if attractions_model:
        all_results.extend(run_inference_unique(attractions_model, image_path, draw, font, processed_classes))
    if landmarks_model:
        all_results.extend(run_inference_unique(landmarks_model, image_path, draw, font, processed_classes))
    if hieroglyph_net_model:
        all_results.extend(run_inference_unique(hieroglyph_net_model, image_path, draw, font, processed_classes))


    unique_results = list(set(all_results))
    st.write("All Detected Unique Classes:", unique_results)

    final_annotated_path = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg").name
    original_image.save(final_annotated_path)
    st.image(final_annotated_path, caption="Final Annotated Image")
    print(unique_results)




