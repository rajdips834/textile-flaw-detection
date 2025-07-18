import torch
import argparse
import os
from data_loader import get_data_loaders
from model import FabricDefectModel
from train import train_model
from evaluate import test
from predict import predict_from_directory
from config import batch_size, learning_rate, epochs

def main(train_model_flag: bool):
    dataset_path = "./dataset"  # Training dataset path
    model_checkpoint_path = "fabric_defect_model_sgd.pth"
    prediction_data_path = "./external_test_data"

    dataset, class_weights, train_loader, val_loader, test_loader = get_data_loaders(dataset_path, batch_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = FabricDefectModel().to(device)
    class_weights = class_weights.to(device)

    if train_model_flag or not os.path.exists(model_checkpoint_path):
        print("\nTraining with SGD Optimizer...")
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
        train_model(model, train_loader, val_loader, optimizer, class_weights, device, epochs, "SGD")

        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'class_weights': class_weights,
            'class_to_idx': dataset.class_to_idx
        }, model_checkpoint_path)

        print("✅ Model saved to:", model_checkpoint_path)
    else:
        print("\n📦 Loading model from checkpoint...")
        checkpoint = torch.load(model_checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        class_weights = checkpoint['class_weights'].to(device)
        print("✅ Model loaded.")

    print("\n🧪 Testing model...")
    test(model, test_loader, class_weights, device)

    print("\n🔍 Predicting from external test directory...")
    prediction = predict_from_directory(model, prediction_data_path, dataset.class_to_idx, device)

    if prediction:
        print(f"\n🎯 Final prediction: {prediction}")
    else:
        print("\n⚠️ No prediction made.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fabric Defect Classifier")
    parser.add_argument('--train', action='store_true', help="Train the model from scratch")
    args = parser.parse_args()

    main(args.train)
