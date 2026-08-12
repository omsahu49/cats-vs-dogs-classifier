import os
import torch
import torchvision.transforms as transforms
from torchvision import models
import streamlit as st
from PIL import Image

# Page setup
st.set_page_config(
    page_title="Cats vs Dogs Classifier",
    page_icon="🐱🐶",
    layout="centered"
)

st.title("🐱🐶 Cats vs Dogs AI Classifier")
st.write("Upload an image to classify whether it's a Cat or a Dog.")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model Loading Function
@st.cache_resource
def load_pytorch_model():
    # List of possible PyTorch weight files
    possible_weights = ["fold_1_model.pth", "model.pth", "best_model.pth", "model.pt"]
    found_weight = None
    
    for w_path in possible_weights:
        if os.path.exists(w_path):
            found_weight = w_path
            break

    # Initialize model (ResNet18 / EfficientNet based on your setup)
    try:
        model = models.resnet18(weights=None)
        num_ftrs = model.fc.in_features
        model.fc = torch.nn.Linear(num_ftrs, 2)
        
        if found_weight:
            model.load_state_dict(torch.load(found_weight, map_location=device))
            model.to(device)
            model.eval()
            return model, None
        else:
            return None, "Model weights file (e.g., `fold_1_model.pth`) not found in GitHub root directory."
    except Exception as e:
        return None, str(e)

model, load_error = load_pytorch_model()

# Image Transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)
    
    if st.button("Classify Image", type="primary"):
        if model is None:
            st.error(f"Failed to load PyTorch model: {load_error}")
        else:
            with st.spinner("Classifying image..."):
                img_tensor = transform(image).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    outputs = model(img_tensor)
                    probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
                    cat_prob = probabilities[0].item() * 100
                    dog_prob = probabilities[1].item() * 100
                
                st.divider()
                if dog_prob > cat_prob:
                    st.success(f"### 🐶 Dog Detected (Confidence: {dog_prob:.2f}%)")
                else:
                    st.success(f"### 🐱 Cat Detected (Confidence: {cat_prob:.2f}%)")
