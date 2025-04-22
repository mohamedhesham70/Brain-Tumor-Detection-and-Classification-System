import tensorflow as tf
import numpy as np
import cv2
import os
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
from ultralytics import YOLO

# Load models
vgg_model = tf.keras.models.load_model(r"D:\Projects\Brain Tumor\VGG16\brain_tumor_vgg16.h5")
yolo_model = YOLO(r"D:\Projects\Brain Tumor\YOLOV8\runs\detect\train\weights\best.pt")

app = Flask(__name__)

# Folder setup
UPLOAD_FOLDER = 'static/uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Labels
CLASS_NAMES = ["No Tumor", "Tumor"]

# Resize helper
def resize_image(image, size=(400, 400)):  # Small size for better display and performance
    return cv2.resize(image, size)

# Preprocess for VGG
def preprocess_image_for_vgg(image_path):
    img = cv2.imread(image_path)
    img = resize_image(img, (224, 224))  # Resize image to fit VGG input
    img = img / 255.0  # Normalize the image
    return np.expand_dims(img, axis=0)

# YOLO detection
def run_yolo_detection(image_path):
    image = cv2.imread(image_path)
    image = resize_image(image, (640, 640))  # YOLO performs well with this size
    results = yolo_model.predict(source=image, imgsz=640, conf=0.5)

    detections = []
    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            name = yolo_model.names[cls_id]
            detections.append({"name": name, "confidence": conf})

    # Annotated image
    annotated = results[0].plot()
    annotated = resize_image(annotated, (400, 400))  # Resize annotated image for display
    output_path = os.path.join(app.config['UPLOAD_FOLDER'], 'detected_' + os.path.basename(image_path))
    cv2.imwrite(output_path, annotated)

    return output_path, detections

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return jsonify({"error": "No file part"})

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"})

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    # Resize uploaded image to 400x400 for better performance
    img = cv2.imread(filepath)
    img = resize_image(img, (400, 400))
    cv2.imwrite(filepath, img)

    return jsonify({"filepath": filepath})

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    image_path = data.get("image_path")

    if not image_path:
        return jsonify({"error": "No image uploaded!"})

    # Classification
    img = preprocess_image_for_vgg(image_path)
    prediction = vgg_model.predict(img)
    classification_result = CLASS_NAMES[int(prediction[0][0] > 0.5)]

    # Detection if tumor exists
    if classification_result == "Tumor":
        detection_image_path, detections = run_yolo_detection(image_path)
    else:
        detection_image_path = None
        detections = []

    return jsonify({
        "classification": classification_result,
        "detection_image": detection_image_path,
        "detections": detections
    })

if __name__ == "__main__":
    app.run(debug=True)
