import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import sys

IMG_SIZE = 224
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

def build_model():
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_features, 256),
        nn.BatchNorm1d(256),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, 1)
    )
    return model

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD)
])

image_path = sys.argv[1]
image = Image.open(image_path).convert('RGB')
tensor = transform(image).unsqueeze(0)

model = build_model()
model.eval()

with torch.no_grad():
    output = torch.sigmoid(model(tensor)).item()

label = "Dog 🐶" if output > 0.5 else "Cat 🐱"
print(f"Prediction: {label} (confidence: {output:.2f})")
