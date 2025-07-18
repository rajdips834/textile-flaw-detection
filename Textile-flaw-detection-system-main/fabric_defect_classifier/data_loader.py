import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch

def get_data_loaders(data_dir, batch_size):
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ])

    dataset = datasets.ImageFolder(root=data_dir, transform=transform)

    class_counts = [0] * len(dataset.classes)
    for _, label in dataset:
        class_counts[label] += 1

    class_weights = 1.0 / torch.tensor(class_counts, dtype=torch.float)
    sample_weights = [class_weights[label] for _, label in dataset]
    sampler = torch.utils.data.WeightedRandomSampler(sample_weights, len(sample_weights))

    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return dataset, class_weights, train_loader, val_loader, test_loader
