import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

def load_dataset(dataset_dir, batch_size=32, img_size=(224, 224), validation_split=0.2, seed=42):
    """
    Loads the dataset from the specified directory using PyTorch.
    """
    if not os.path.exists(dataset_dir):
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    print(f"Loading dataset from: {dataset_dir}")

    # Define transforms (Phishpedia/ResNet style)
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    full_dataset = datasets.ImageFolder(root=dataset_dir, transform=transform)
    class_names = full_dataset.classes
    print(f"Classes: {class_names}")

    # Split dataset
    dataset_size = len(full_dataset)
    val_size = int(validation_split * dataset_size)
    train_size = dataset_size - val_size

    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=generator)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    return train_loader, val_loader, class_names

if __name__ == "__main__":
    # Test dataset loading
    dataset_path = "Dataset"
    train_loader, val_loader, classes = load_dataset(dataset_path)
    for images, labels in train_loader:
        print(f"Batch image shape: {images.shape}")
        print(f"Batch label shape: {labels.shape}")
        break
