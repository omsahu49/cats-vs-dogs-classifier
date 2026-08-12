import os
import torch
import torchvision.transforms as transforms
from torchvision import models
import streamlit as st
from PIL import Image

# Page Configuration
st.set_page_config(
    page_title="Cats vs Dogs Classifier",
    page_icon="🐱🐶",
    layout="wide"
)

# Custom Gradio Dark Theme Styling
st.markdown("""
<style>
    .stApp {
        background-color: #0b0f19;
        color: #ffffff;
    }
    div[data-testid="column"] {
        background-color: #121826;
        border: 1px solid #1f293d;
        border-radius: 8px;
        padding: 18px;
    }
    .stButton>button {
        border-radius: 6px;
        font-weight: 600;
        height: 45px;
    }
    div[data-testid="column"]:nth-child(1) .stButton>button[kind="primary"] {
        background-color: #ff5500;
        color: white;
        border: none;
    }
    div[data-testid="column"]:nth-child(1) .stButton>button[kind="primary"]:hover {
        background-color: #e04b00;
    }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

st.markdown("## 🐱🐶 Cats vs Dogs AI Classifier")
st.markdown("<p style='color: #9ca3af;'>ResNet50 Transfer Learning pet classification model.</p>", unsafe_allow_html=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource
def load_pytorch_model():
    # Checking for any .pth / .pt file in the repo root
    files_in_dir = os.listdir(".")
    pth_files = [f for f in files_in_dir if f.endswith(".pth") or f.endswith(".pt")]
    
    found_weight = None
    if pth_files:
        found_weight = os.path.abspath(pth_files[0])

    try:
        # Using ResNet50 as specified in your training pipeline
        model = models.resnet50(weights=None)
        num_ftrs = model.fc.in_features
        model.fc = torch.nn.Linear(num_ftrs, 2)
        
        if found_weight:
            state_dict = torch.load(found_weight, map_location=device)
            # Support both raw state_dict and dicts with 'model' or 'state_dict' keys
            if "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]
            elif "model" in state_dict:
                state_dict = state_dict["model"]
                
            model.load_state_dict(state_dict, strict=False)
            model.to(device)
            model.eval()
            return model, None, os.path.basename(found_weight)
        else:
            return None, "No `.pth` or `.pt` weights file found in GitHub repo. Please upload your model weights file.", None
    except Exception as e:
        return None, str(e), None

model, load_error, weight_filename = load_pytorch_model()

# ResNet Image Preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

col1, col2 = st.columns([1, 1], gap="medium")

with col1:
    st.markdown("<h5 style='color: #9ca3af;'>🖼️ Upload Image</h5>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png", "webp"], label_visibility="collapsed")
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, use_container_width=True)
    
    b_col1, b_col2 = st.columns([1, 1])
    with b_col1:
        clear_btn = st.button("Clear", use_container_width=True)
    with b_col2:
        submit_btn = st.button("Submit", type="primary", use_container_width=True)

with col2:
    st.markdown("<h5 style='color: #9ca3af;'>📊 Classification Result</h5>", unsafe_allow_html=True)
    
    if uploaded_file is not None and submit_btn:
        if model is None:
            st.error(f"Failed to load PyTorch model: {load_error}")
        else:
            with st.spinner("Analyzing image through ResNet50..."):
                img_tensor = transform(image).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    outputs = model(img_tensor)
                    probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
                    cat_prob = probabilities[0].item() * 100
                    dog_prob = probabilities[1].item() * 100
                
                st.markdown("---")
                if dog_prob > cat_prob:
                    st.success(f"### 🐶 Status: DOG DETECTED\n**Confidence:** {dog_prob:.2f}%")
                else:
                    st.success(f"### 🐱 Status: CAT DETECTED\n**Confidence:** {cat_prob:.2f}%")
                
                st.markdown("<h5 style='color: #9ca3af; margin-top:20px;'>Probability Breakdown</h5>", unsafe_allow_html=True)
                st.write(f"🐱 **Cat Probability:** {cat_prob:.2f}%")
                st.progress(int(cat_prob))
                st.write(f"🐶 **Dog Probability:** {dog_prob:.2f}%")
                st.progress(int(dog_prob))
    else:
        if weight_filename:
            st.info(f"Loaded weights file: `{weight_filename}`. Upload an image on the left and click 'Submit'.")
        else:
            st.warning("⚠️ Weights file missing in repository! Upload your `.pth` file to GitHub.")

    st.markdown("<br>", unsafe_allow_html=True)
    st.button("🔗 Share via Link", disabled=True, use_container_width=True)
