import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from dataset_loader import load_dataset
import numpy as np
from sklearn.utils.class_weight import compute_class_weight

def build_model(num_classes=2):
    # Load ResNet50 (similar to Phishpedia's Siamese backbone)
    print("Building PyTorch ResNet50 model...")
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    
    # Freeze earlier layers
    for param in model.parameters():
        param.requires_grad = False
        
    # Unfreeze the last block (layer4) and fc for fine-tuning
    for param in model.layer4.parameters():
        param.requires_grad = True

    # Replace classification head
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_ftrs, num_classes)
    )
    return model

def calculate_class_weights(dataset_dir, classes, penalty_factor=2.0):
    """
    Computes class weights to aggressively penalize False Negatives.
    """
    labels = []
    class_indices = {cls: idx for idx, cls in enumerate(classes)}
    
    for cls in classes:
        cls_dir = os.path.join(dataset_dir, cls)
        count = len([name for name in os.listdir(cls_dir) if os.path.isfile(os.path.join(cls_dir, name))])
        labels.extend([class_indices[cls]] * count)
        
    labels = np.array(labels)
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(labels), y=labels)
    
    # Assume 'phishing' is the first class due to alphabetical ordering
    if classes[0] == 'phishing':
        class_weights[0] *= penalty_factor
        
    print(f"Computed Class Weights: {class_weights}")
    return torch.tensor(class_weights, dtype=torch.float32)

def train():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataset_dir = "Dataset"
    train_loader, val_loader, class_names = load_dataset(dataset_dir, batch_size=32)

    model = build_model(num_classes=len(class_names))
    model = model.to(device)

    # Class Weights for Loss
    class_weights = calculate_class_weights(dataset_dir, class_names, penalty_factor=2.0)
    class_weights = class_weights.to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training Loop
    epochs = 2
    best_val_loss = float('inf')
    patience = 5
    patience_counter = 0

    os.makedirs("model", exist_ok=True)
    model_save_path = "model/fake_login_detector.pth"

    for epoch in range(epochs):
        # TRAIN
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        train_loss = running_loss / len(train_loader.dataset)
        train_acc = correct / total

        # VALIDATION
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
        val_loss = val_loss / len(val_loader.dataset)
        val_acc = val_correct / val_total

        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} - Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        # Early Stopping & Checkpointing
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), model_save_path)
            print("Saved new best model.")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

    print("Training complete. Best model saved to", model_save_path)

if __name__ == "__main__":
    train()
