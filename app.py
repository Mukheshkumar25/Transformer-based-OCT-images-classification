import streamlit as st
import torch
from PIL import Image
from torchvision import transforms
from mobilevit_model import MobileViT_XS
import torch.nn.functional as F

# Title and description
st.title("Retinal OCT Disease Classification")
st.write("Upload an OCT image to classify it into CNV, DME, DRUSEN, or NORMAL using MobileViT-XS.")

# -----------------------------
# Load Model
# -----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

model = MobileViT_XS(num_classes=4).to(device)
checkpoint = torch.load("mobilevit_xs_best.pth", map_location=device, weights_only=False)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

idx_to_class = {0:"CNV", 1:"DME", 2:"DRUSEN", 3:"NORMAL"}

# -----------------------------
# Image Transform
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor()
])

# -----------------------------
# File Upload UI
# -----------------------------
uploaded_file = st.file_uploader("Upload OCT Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    
    # Display uploaded image
    st.image(img, caption="Uploaded OCT Image", width=300)

    # Preprocess
    x = transform(img).unsqueeze(0).to(device)

    # Prediction
    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1)
        pred_idx = probs.argmax(dim=1).item()
        confidence = probs[0][pred_idx].item()

    # Display prediction
    st.subheader(f"Prediction: **{idx_to_class[pred_idx]}**")
    st.write(f"Confidence: **{confidence * 100:.2f}%**")
