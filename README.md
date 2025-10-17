# 🕊️ Aerial Object Classification & Detection

## 📘 Overview
This project focuses on building a **Deep Learning-based system** capable of classifying aerial images into **Bird** or **Drone**, and optionally performing **Object Detection** using **YOLOv8**.  
It supports applications in **Aerial Surveillance**, **Wildlife Monitoring**, and **Security & Defense** sectors where accurate distinction between drones and birds is crucial.

---

## 🎯 Project Objectives
- Build a **Custom CNN** and **Transfer Learning-based model** for aerial object classification.  
- Implement **YOLOv8** for real-time object detection (optional).  
- Deploy the final solution via **Streamlit** for interactive prediction.  

---

## 🧠 Skills You’ll Gain
- Deep Learning  
- Computer Vision  
- Image Classification & Object Detection  
- Python (TensorFlow/Keras or PyTorch)  
- Data Preprocessing & Augmentation  
- Model Evaluation & Optimization  
- Streamlit Deployment  

---

## 🌍 Domain
**Aerial Surveillance | Wildlife Monitoring | Security & Defense Applications**

---

## 🧩 Problem Statement
The goal is to build an AI-based solution that can:
- **Classify** aerial images into *Bird* or *Drone*.  
- **Detect & localize** these objects in real-world aerial footage (optional YOLOv8 extension).  

This aids in:
- Airspace safety (detect unauthorized drones).  
- Wildlife protection (avoid bird strikes).  
- Automated aerial monitoring and environmental research.

---

## 💼 Real-World Applications

### 🦅 Wildlife Protection
Detect birds near wind farms or airports to prevent accidents.

### 🛰️ Security & Defense
Identify drones in restricted or sensitive zones for immediate alerts.

### ✈️ Airport Bird-Strike Prevention
Monitor runways for bird activity to reduce flight hazards.

### 🌿 Environmental Research
Track and analyze bird populations using aerial imagery.

---

## ⚙️ Project Workflow

### 1️⃣ Dataset Understanding
- Explore folder structure.  
- Count images per class.  
- Identify class imbalance.  
- Visualize sample images.

### 2️⃣ Data Preprocessing
- Normalize pixel values to `[0, 1]`.  
- Resize all images to a fixed dimension (e.g., 224×224).  

### 3️⃣ Data Augmentation
- Apply transformations: rotation, flipping, zoom, brightness variation, cropping.

### 4️⃣ Model Building (Classification)
- **Custom CNN:** Convolutional, pooling, dropout, batch normalization, and dense layers.  
- **Transfer Learning:** Fine-tune pre-trained models like *ResNet50*, *MobileNet*, *EfficientNetB0*.

### 5️⃣ Model Training
- Use **EarlyStopping** and **ModelCheckpoint**.  
- Track **Accuracy**, **Precision**, **Recall**, and **F1-Score**.

### 6️⃣ Model Evaluation
- Evaluate on test data using a **confusion matrix** and **classification report**.  
- Visualize **accuracy** and **loss curves**.

### 7️⃣ Model Comparison
- Compare results of custom CNN vs transfer learning models.  
- Select the best-performing model for deployment.

---

## 🧾 Optional Module: Object Detection (YOLOv8)

### Steps:
1. Install YOLOv8 (`pip install ultralytics`)  
2. Prepare dataset (images + YOLOv8-format `.txt` label files).  
3. Create `data.yaml` configuration file.  
4. Train YOLOv8 model using the command:  
   ```bash
   yolo train model=yolov8n.pt data=data.yaml epochs=50 imgsz=640
