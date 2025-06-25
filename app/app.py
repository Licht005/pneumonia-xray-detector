import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import os
import sys
import json

# Add src directory to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from model import get_resnet18_model

# Set up page
st.set_page_config(page_title="Pneumonia Detector", layout="wide")
st.title("🫁 Pneumonia Detection from Chest X-rays")
st.markdown("Upload a chest X-ray to check for **NORMAL** or **PNEUMONIA**. The model may also estimate the likely type.")

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = get_resnet18_model()
model_path = "data/pneumonia_model.pth"

try:
    model.load_state_dict(torch.load(model_path, map_location=device))
except FileNotFoundError:
    st.error(f"Model file not found at `{model_path}`.")
    st.stop()

model.eval()
model.to(device)

# Transform for input
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485], [0.229])  # Grayscale normalization
])

# File uploader
uploaded_file = st.file_uploader("📤 Upload a Chest X-ray Image", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("L")
    st.image(image, caption="Uploaded Image", width=350)

    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        confidence = torch.max(probabilities).item()
        pred_index = torch.argmax(probabilities).item()
        classes = ["NORMAL", "PNEUMONIA"]
        prediction = classes[pred_index]

    st.subheader("🧪 Prediction Result")
    st.write(f"**Prediction:** `{prediction}`")
    st.write(f"**Confidence Score:** `{confidence * 100:.2f}%`")

    est_type = "N/A"
    if prediction == "PNEUMONIA":
        est_type = "Viral" if confidence < 0.9 else "Bacterial"
        st.write(f"**Estimated Type:** `{est_type} Pneumonia`")

    st.markdown("🩻 *Grad-CAM visualization coming soon.*")

    # Download button
    result_data = {
        "filename": uploaded_file.name,
        "prediction": prediction,
        "confidence": round(confidence * 100, 2),
        "estimated_type": est_type
    }
    st.download_button(
        label="📥 Download Prediction Result",
        data=json.dumps(result_data, indent=2),
        file_name=f"{uploaded_file.name}_result.json",
        mime="application/json"
    )

# Sidebar (info only)
st.sidebar.markdown("---")
st.sidebar.markdown("### ℹ️ Pneumonia Info")
st.sidebar.info(
    "Pneumonia is a lung infection that can be viral or bacterial. "
    "X-ray screening is a supportive tool and should be followed by professional diagnosis."
)
