# Brain-Tumor-Detection-and-Classification-System

🛠️ Tools Used
Deep Learning Frameworks:
TensorFlow, Keras, Ultralytics (YOLOv8)

Backend & Deployment:
Flask (API-like endpoints), HTML/CSS/JavaScript (Frontend)

Image Processing:
PIL

Data Augmentation & Training:
ImageDataGenerator

Model Architectures:
VGG16 (Classification), YOLOv8 (Object Detection)

Additional Tools:
NumPy, Pandas, Matplotlib, Seaborn

🧠 Project Description
Developed an end-to-end brain tumor diagnosis system using MRI scans, combining classification and object detection to support medical decision-making.

🔍 VGG16 – Tumor Classification
Trained a VGG16-based CNN to classify MRI images into "Tumor" or "No Tumor"
→ Achieved 98% accuracy on test data.

Integrated data augmentation techniques: rotation, flipping, shifting.

Deployed via Flask: Processes user-uploaded images and returns immediate classification results.

🎯 YOLOv8 – Tumor Detection & Subtype Classification
Fine-tuned YOLOv8n on a custom-labeled dataset to:

Localize tumors.

Classify into Glioma, Meningioma, or Pituitary Tumor.

Results:

92.2% mAP50

71.3% mAP50-95 (on validation data)

Real-time bounding box visualizations for medical interpretability.

🧩 Integrated Flask Application (app.py)
Built a smart pipeline:

If image is "No Tumor" → return result immediately.

If image is "Tumor" → trigger YOLOv8 to detect tumor location & subtype.

Designed a user-friendly interface using HTML/CSS/JavaScript with dynamic result rendering.

🌟 Key Achievements
✅ Reduced false positives by cascading models:
VGG16 (first-pass) → YOLOv8 (detailed detection)

✅ Enhanced interpretability with visual bounding boxes for clinical insights.

✅ Optimized inference speed to < 100ms per prediction for real-time usage.

⚙️ Technical Highlights
Model Optimization:
Used transfer learning, early stopping, and dropout layers to reduce overfitting.

Data Pipeline:
Automated dataset splitting (70% Train, 15% Validation, 15% Test) and preprocessing.

Scalability:
Modular code design allows easy integration of future models (e.g., YOLOv9).

