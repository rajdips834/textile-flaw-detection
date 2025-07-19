# data_loader.py

import os
from collections import Counter
from PIL import Image
import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, WeightedRandomSampler
from config import image_size, batch_size

def get_data_loaders(data_dir):
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor()
    ])

    dataset = datasets.ImageFolder(root=data_dir, transform=transform)

    # Split dataset
    total_size = len(dataset)
    train_size = int(0.7 * total_size)
    val_size = int(0.15 * total_size)
    test_size = total_size - train_size - val_size
    train_data, val_data, test_data = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])

    # Compute sampler from train data only
    train_labels = [dataset[i][1] for i in train_data.indices]
    class_counts = Counter(train_labels)
    print("Train class distribution:", class_counts)

    class_weights = torch.tensor([
        1.0 / class_counts[i] if class_counts[i] > 0 else 0.0
        for i in range(len(dataset.classes))
    ], dtype=torch.float)

    sample_weights = [class_weights[label] for label in train_labels]
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(train_data, batch_size=batch_size, sampler=sampler)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    return dataset, class_weights, train_loader, val_loader, test_loader
