import os
from PIL import Image
from torchvision import transforms
from collections import Counter
import torch

def process_patches(model, image_paths, class_to_idx, device):
    patch_predictions = []
    print("\nPatch-by-patch predictions:\n")

    for image_path in image_paths:
        try:
            patch = Image.open(image_path).convert('L')
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])
            patch_tensor = transform(patch).unsqueeze(0).to(device)

            with torch.no_grad():
                output = model(patch_tensor)
                _, predicted_index = torch.max(output, 1)
                predicted_label = list(class_to_idx.keys())[list(class_to_idx.values()).index(predicted_index.item())]
                patch_predictions.append(predicted_label)
                print(f"{os.path.basename(image_path)} → Predicted class: {predicted_label}")

        except Exception as e:
            print(f"Error processing {image_path}: {e}")

    prediction_counts = Counter(patch_predictions)
    most_common = prediction_counts.most_common(1)[0][0] if prediction_counts else None
    return most_common

def predict_from_directory(model, directory, class_to_idx, device, num_patches=64):
    image_paths = [os.path.join(directory, f) for f in os.listdir(directory)
                   if f.lower().endswith(('.png', '.jpg', '.jpeg'))][:num_patches]

    if not image_paths:
        print("No images found in the specified directory.")
        return None

    return process_patches(model, image_paths, class_to_idx, device)
