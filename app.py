import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
import gradio as gr
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model Architecture Setup & Weights Load
def load_model():
    model = models.resnet50(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    
    try:
        state_dict = torch.load("fold_1.pth", map_location=device)
        model.load_state_dict(state_dict)
    except Exception as e:
        print(f"Error loading model weights: {e}")
        
    model = model.to(device)
    model.eval()
    return model

model = load_model()

# Image Preprocessing Transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

classes = ["Cat 🐱", "Dog 🐶"]

def predict_image(image):
    if image is None:
        return "Please upload an image!"
    
    img = Image.fromarray(image).convert('RGB')
    tensor_img = transform(img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(tensor_img)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        
    confidences = {classes[i]: float(probabilities[i]) for i in range(2)}
    return confidences

interface = gr.Interface(
    fn=predict_image,
    inputs=gr.Image(label="Upload Image"),
    outputs=gr.Label(num_top_classes=2, label="Prediction Confidence"),
    title="🏆 Cats vs Dogs Image Classifier",
    description="ResNet50 Transfer Learning Model trained with K-Fold Cross Validation."
)

if __name__ == "__main__":
    interface.launch()