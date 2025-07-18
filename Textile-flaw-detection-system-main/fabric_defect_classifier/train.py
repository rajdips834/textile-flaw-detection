import torch


def train_model(model, train_loader, val_loader, optimizer, class_weights, device, epochs, optimizer_name):
    history = []
    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            images, labels = batch
            images, labels = images.to(device), labels.to(device)
            loss = model.training_step((images, labels), class_weights)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        result = evaluate(model, val_loader, class_weights, device)
        model.epoch_end(epoch, result, optimizer_name)
        history.append(result)
    return history

def evaluate(model, val_loader, class_weights, device):
    model.eval()
    outputs = []
    with torch.no_grad():
        for batch in val_loader:
            images, labels = batch
            images, labels = images.to(device), labels.to(device)
            outputs.append(model.validation_step((images, labels), class_weights))
    return model.validation_epoch_end(outputs)
