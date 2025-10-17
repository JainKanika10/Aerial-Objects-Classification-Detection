import streamlit as st
from ultralytics import YOLO
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms
import tempfile, os

# -------------------------------
# 1️⃣ Streamlit Setup
# -------------------------------
st.set_page_config(
    page_title="Aerial Object Classifier & Detector",
    page_icon="🕊️",
    layout="wide"
)

# -------------------------------
# 2️⃣ Custom CSS for colorful background
# -------------------------------
st.markdown(
    """
    <style>
    /* Gradient background */
    .stApp {
        background: linear-gradient(to bottom right, #a1c4fd, #c2e9fb, #fbc2eb);
        background-attachment: fixed;
        color: #000;
    }

    /* Centered title */
    .title {
        text-align: center;
        font-size: 42px;
        font-weight: bold;
        color: #1f2f5c;
    }

    /* Card style for metrics */
    .metric-card {
        background-color: rgba(255, 255, 255, 0.8);
        padding: 15px;
        border-radius: 15px;
        text-align: center;
    }

    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<h1 class="title">🕊️ Aerial Object Classification & Detection</h1>', unsafe_allow_html=True)
st.markdown(
    '<p style="text-align:center;font-size:18px;">Upload an image to classify as <b>Bird/Drone</b> or detect objects using YOLOv8.</p>',
    unsafe_allow_html=True
)

# -------------------------------
# 3️⃣ Sidebar Options
# -------------------------------
st.sidebar.header("Settings")
mode = st.sidebar.radio("Select Mode", ["Classification (ResNet50)", "Detection (YOLOv8)"])
st.sidebar.markdown("---")
st.sidebar.info("ℹ️ Upload an image (jpg, jpeg, png) and choose a mode to analyze it.")

# -------------------------------
# 4️⃣ Load Models (cached)
# -------------------------------
@st.cache_resource
def load_resnet_model():
    model = models.resnet50(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False

    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_features, 128),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(128, 1),
        nn.Sigmoid()
    )
    model.load_state_dict(torch.load('best_model.pth', map_location=torch.device('cpu')))
    model.eval()
    return model

@st.cache_resource
def load_yolo_model():
    model = YOLO("yolov8n.pt")  # replace with your trained YOLO model
    return model

# -------------------------------
# 5️⃣ Image Upload
# -------------------------------
uploaded_file = st.file_uploader("📤 Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)
    st.markdown("---")

    # -------------------------------
    # 🧩 Classification Mode
    # -------------------------------
    if mode == "Classification (ResNet50)":
        st.info("Running classification... please wait ⏳")
        model = load_resnet_model()

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        img_tensor = transform(image).unsqueeze(0)

        with torch.no_grad():
            output = model(img_tensor)
            prob = output.item()
            pred_class = "🚁 Drone" if prob > 0.5 else "🕊️ Bird"
            confidence = prob if prob > 0.5 else (1 - prob)

        # Display result in a colorful card
        st.markdown(
            f"""
            <div class="metric-card">
                <h2>Prediction: {pred_class}</h2>
                <h3>Confidence: {confidence*100:.2f}%</h3>
            </div>
            """,
            unsafe_allow_html=True
        )
        st.progress(confidence)

    # -------------------------------
    # 🔍 Detection Mode
    # -------------------------------
    else:
        st.info("Running YOLOv8 detection... please wait ⏳")
        model = load_yolo_model()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            image.save(tmp.name)
            temp_path = tmp.name

        results = model.predict(source=temp_path, conf=0.25)
        result = results[0]

        if len(result.boxes) > 0:
            st.success(f"✅ Detected {len(result.boxes)} object(s):")
            for i, box in enumerate(result.boxes):
                cls_id = int(box.cls)
                conf = float(box.conf)
                label = model.names[cls_id]
                st.markdown(f"**Object {i+1}:** {label} | Confidence: {conf:.2f}")

            # Display detection image
            result_img = result.plot()
            st.image(result_img, caption="Detection Results", use_container_width=True, channels="BGR")
        else:
            st.warning("No objects detected.")

        os.remove(temp_path)
