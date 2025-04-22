# Brain-Tumor-Detection-and-Classification-System

## 🧠 Brain Tumor Detection and Classification System

This project presents a powerful **AI-based system** that leverages **two deep learning models — VGG16 & YOLOv8** — to **classify, detect, and localize brain tumors** from MRI images.

---

## 🔍 System Pipeline Overview

The system follows a **two-stage workflow**:

1. 🎯 **Classification (VGG16)** – Determines if a brain tumor exists (Binary classification: Tumor / No Tumor).
2. 🎯 **Detection & Typing (YOLOv8)** – Locates the tumor and classifies it as:
   - Glioma
   - Meningioma
   - Pituitary

---

## ✅ Stage 1: Tumor Classification using VGG16

### 🔸 Purpose:
- Binary classification of MRI images.

### 🔸 Highlights:
- Transfer learning from ImageNet.
- Final layers fine-tuned for tumor detection.
- Achieved **97.56% accuracy** on test data.

### 🔸 Model Architecture:
```python
Sequential([
    base_model,  # VGG16 without top layers
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])
```

### 🔸 Training Details:
- Optimizer: Adam  
- Loss: Binary Crossentropy  
- Early Stopping: Enabled (patience=5)  
- Max Epochs: 20

---

## ✅ Stage 2: Tumor Detection & Classification using YOLOv8

### 🔸 Purpose:
- Localize and classify tumor types with bounding boxes.

### 🔸 Model Info:
- Pre-trained **YOLOv8n**, fine-tuned on MRI tumor dataset.
- Training:
  - Epochs: 70
  - Optimizer: AdamW
  - LR: 0.001
  - Early stopping: patience=15

### 🔸 Performance:
- **mAP50: 92.1%**
  - Glioma: 83.4%
  - Meningioma: 97.7%
  - Pituitary: 95.3%

---

## 💻 Web App Integration (Flask-Based)

### 🧭 Workflow:
1. User uploads an MRI image.
2. VGG16 processes the image:
   - **No Tumor?** → Result shown.
   - **Tumor?** → Image sent to YOLOv8.
3. YOLOv8:
   - Draws bounding boxes.
   - Classifies tumor type.
4. Frontend displays results.

### 🛠️ Components:
- `app.py`: Flask backend (file upload + model integration)
- `index.html`: Frontend UI (clean and responsive)

---

## 📋 Example Output

- ✅ Prediction: Tumor Detected  
- 🧠 Type: Meningioma  
- 📍 Bounding Box: Shown on image  
- 📊 Confidence: 97.7%

---

## ⚙️ Tech Stack

- Python 3.x  
- TensorFlow / Keras  
- Ultralytics (YOLOv8)  
- OpenCV  
- Flask  
- Required libs: `requirements.txt`

---

## 🎯 Final Note

This system offers a **powerful AI solution for brain tumor analysis**, combining:
- **Speed** (real-time detection),
- **Accuracy** (97%+ classification, 92%+ detection),
- and **Visualization** (bounding boxes + tumor types).
