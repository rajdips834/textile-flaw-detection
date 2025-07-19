# main.py

import torch
import argparse
from config import batch_size, learning_rate, epochs
from data_loader import get_data_loaders
from model import SimpleCNN
from trainer import train_model
from evaluator import test
from predictor import predict_from_directory

def get_device():
    if torch.backends.mps.is_available():
        print("Using Apple MPS (Metal) backend for GPU acceleration.")
        return torch.device("mps")
    elif torch.cuda.is_available():
        print("Using CUDA GPU.")
        return torch.device("cuda")
    else:
        print("Using CPU.")
        return torch.device("cpu")

def main(train_flag: bool):
    # Dataset path
    dataset_path = "./dataset"
    
    # Load data
    dataset, class_weights, train_loader, val_loader, test_loader = get_data_loaders(dataset_path)
    device = get_device()

    # Initialize model
    model = SimpleCNN().to(device)
    class_weights = class_weights.to(device)

    # Train or load model
    if train_flag:
        print("\nTraining with SGD Optimizer...")
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
        train_model(model, train_loader, val_loader, optimizer, class_weights, device, epochs)

        # Save model
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'class_weights': class_weights,
            'class_to_idx': dataset.class_to_idx
        }, 'fabric_defect_model_sgd.pth')
    else:
        # Load pre-trained model
        print("Loading saved model weights...")
        checkpoint = torch.load('fabric_defect_model_sgd.pth', map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        class_weights = checkpoint['class_weights'].to(device)
        dataset.class_to_idx = checkpoint['class_to_idx']

    # Evaluate model
    test(model, test_loader, class_weights, device)

    # Predict from external data
    print("\nPredicting from external test images...")
    predictions = predict_from_directory(model, "./external_test_data", dataset.class_to_idx, device)

    if predictions:
        print("\nFinal Predictions:")
        for file, pred in predictions:
            print(f"{file}: {pred}")
    else:
        print("\nNo predictions made.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fabric Defect Classifier")        
    parser.add_argument('--train', action='store_true', help="Train the model before testing")
    args = parser.parse_args()

    main(train_flag=args.train)
