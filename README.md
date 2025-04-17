# Brain-Tumor-Detection-and-Classification-System

Tools:

Deep Learning Frameworks: TensorFlow, Keras, Ultralytics (YOLOv8)

Backend & Deployment: Flask (API-like endpoints), HTML/CSS/JavaScript (Frontend)  

Image Processing: PIL

Data Augmentation & Training: ImageDataGenerator

Model Architectures: VGG16 (Classification), YOLOv8 (Object Detection)

Additional Tools: NumPy, Pandas, Matplotlib, Seaborn

Description:
Developed an end-to-end system for brain tumor diagnosis using MRI scans, combining classification and object detection models to enhance medical decision-making:

VGG16 Classification Model

Trained a VGG16-based CNN to classify MRI images into "Tumor" or "No Tumor" with 98% accuracy on test data.

Integrated data augmentation (rotation, flipping, shifting) to improve generalization.

Deployed via Flask to process user-uploaded images and return immediate binary results.

YOLOv8 Tumor Detection & Subtype Classification

Fine-tuned YOLOv8n on a custom dataset to localize tumors and classify them into three subtypes:

Glioma | Meningioma | Pituitary Tumor

Achieved 92.2% mAP50 and 71.3% mAP50-95 on validation data, with real-time bounding box visualization.

Integrated Flask Application (app.py)

Built a pipeline where VGG16 acts as a first-stage filter:

If "No Tumor", returns immediate result.

If "Tumor", triggers YOLOv8 to pinpoint tumor location and identify subtype.

Designed a user-friendly interface with dynamic result displays using HTML/CSS/JavaScript.

Key Achievements:
Reduced false positives by cascading models (VGG16 first-pass → YOLOv8 detailed analysis).

Enabled interpretability with bounding box visualizations for clinical use cases.

Optimized model inference speed (under 100ms per prediction) for practical deployment.

Technical Highlights:
Model Optimization: Used transfer learning, early stopping, and dropout layers to prevent overfitting.

Data Pipeline: Automated dataset splitting (70% train, 15% validation, 15% test) and preprocessing.

Scalability: Modular code structure allows easy integration of newer models (e.g., YOLOv9).
