from ultralytics import YOLO
import sys
import os

model = YOLO('cdw_best_model.pt')

image_path = sys.argv[1] if len(sys.argv) > 1 else 'test.jpg'

results = model.predict(image_path, conf=0.5, save=True)

print(f"Detection complete! Results saved in 'runs/detect/' folder")
