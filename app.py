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
    /* Dark Background Override */
    .stApp {
        background-color: #0b0f19;
        color: #ffffff;
    }
    
    /* Box Container styling like Gradio */
    div[data-testid="column"] {
        background-color: #121826;
        border: 1px solid #1f293d;
        border-radius: 8px;
        padding: 18px;
    }

    /* Buttons Styling */
    .stButton>button {
        border-radius: 6px;
        font-weight: 600;
        height: 45px;
    }
    
    /* Primary Submit Button (Gradio Orange) */
    div[data-testid="column"]:nth-child(1) .stButton>button[kind="primary"] {
        background-color: #ff5500;
        color: white;
        border: none;
    }
    div[data-testid="column"]:nth-child(1) .stButton>button[kind="primary"]:hover {
        background-color: #e04b00;
    }

    /* Hide Streamlit Header & Footer */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# Title & Subtitle
st.markdown("## 🐱🐶 Cats vs Dogs AI Classifier")
st.markdown("<p style='color: #9ca3af;'>Deep Learning ResNet Vision Model for pet classification.</p>", unsafe_allow_html=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource
def load_pytorch_model():
    possible_weights = ["fold_1_model.pth", "model.pth", "best_model.pth", "model.pt"]
    found_weight = None
    
    for w_path in possible_weights:
        if os.path.exists(w_path):
            found_weight = w_path
            break

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
            return None, f"Model weights file (e.g. `fold_1_model.pth`) not found in repo root."
    except Exception as e:
        return None, str(e)

model, load_error = load_pytorch_model()

# Transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 2 Column Side-by-Side Gradio Layout
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
            with st.spinner("Analyzing image..."):
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
        st.markdown("<div style='height: 250px; display: flex; align-items: center; justify-content: center; border: 1px dashed #2e384d; border-radius: 6px; color: #6b7280;'>Results will appear here after clicking Submit</div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.button("🔗 Share via Link", disabled=True, use_container_width=True)
