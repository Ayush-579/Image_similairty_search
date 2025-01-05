import argparse
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from Autoencoder import Autoencoder, train_autoencoder
from Resnet50 import ResNet50
from VGG16 import VGG16
from Vit_b import ViT
from evaluate import evaluate_model

def load_dataset(batch_size=32, train=True, shuffle=False):
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Load either training or testing dataset
    dataset = datasets.FashionMNIST(root='./data', train=train, download=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader

def save_models(models):
    for model_name, model in models.items():
        torch.save(model.state_dict(), f"{model_name.lower()}.pth")
        print(f"{model_name} saved as {model_name.lower()}.pth")

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Evaluate image similarity models")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for dataloader")
    parser.add_argument("--top_k", type=int, default=20, help="Top-K value for evaluation metrics")
    parser.add_argument("--save_models", action="store_true", help="Save models after initialization")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs for training the Autoencoder")
    args = parser.parse_args()

    # Load training and test dataset
    train_dataloader = load_dataset(batch_size=args.batch_size, train=True, shuffle=True)
    test_dataloader = load_dataset(batch_size=args.batch_size, train=False, shuffle=False)

    # Define models to evaluate
    models = {
        "Autoencoder": Autoencoder(),
        "ResNet50": ResNet50(),
        "VGG16": VGG16(),
        "ViT": ViT()
    }

    # Save models if flag is passed
    if args.save_models:
        save_models(models)

    # Evaluate each model
    for model_name, model in models.items():
        print(f"Evaluating {model_name}...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

        # Train Autoencoder specifically
        if model_name == "Autoencoder":
            print("Training Autoencoder...")
            model = train_autoencoder(model, train_dataloader, epochs=args.epochs)
            torch.save(model.state_dict(), f"{model_name.lower()}.pth")  # Save trained Autoencoder
       
        else:
            try:
                model.load_state_dict(torch.load(f"{model_name.lower()}.pth"))  # Load pretrained weights
            except FileNotFoundError:
                print(f"Pretrained model for {model_name} not found. Training from scratch.")
                model = model.to(device)  # Initialize with random weights if not pretrained
            model.eval()

        # Evaluate the model
        precision, recall, f1, accuracy, lrap = evaluate_model(model, test_dataloader, top_k=args.top_k)

        # Display results
        print(f"Results for {model_name}:")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1-Score: {f1:.4f}")
        print(f"  LRAP: {lrap:.4f}")
        print(f"  Accuracy: {accuracy:.4f}")
        print("-" * 40)

if __name__ == "__main__":
    main()
