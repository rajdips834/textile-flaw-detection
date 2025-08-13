# trainer.py

import torch
import torch.nn as nn
from evaluator import evaluate

def train_model(model, train_loader, val_loader, optimizer, class_weights, device, num_epochs):
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss / total
        train_accuracy = correct / total * 100
        val_loss, val_accuracy = evaluate(model, val_loader, class_weights, device)

        print(f"Epoch [{epoch+1}/{num_epochs}] "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}% "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%")
