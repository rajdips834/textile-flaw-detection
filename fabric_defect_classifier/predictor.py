# predictor.py

import os
from PIL import Image
import torch
from torchvision import transforms
from config import image_size

def predict_from_directory(model, directory_path, class_to_idx, device):
    model.eval()
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor()
    ])

    idx_to_class = {v: k for k, v in class_to_idx.items()}
    predictions = []

    if not os.path.exists(directory_path):
        print(f"Directory {directory_path} does not exist.")
        return []

    files = [f for f in os.listdir(directory_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not files:
        print("No images found in the specified directory.")
        return []

    for file in files:
        image_path = os.path.join(directory_path, file)
        try:
            image = Image.open(image_path).convert("RGB")
            image = transform(image).unsqueeze(0).to(device)

            with torch.no_grad():
                output = model(image)
                pred_class_idx = output.argmax(dim=1).item()
                pred_class = idx_to_class[pred_class_idx]
                predictions.append((file, pred_class))
                print(f"{file}: {pred_class}")
        except Exception as e:
            print(f"Failed to process {file}: {e}")

    return predictions
