import os
import torch
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, f1_score
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from dataset_loader import load_dataset
from train import build_model

def evaluate():
    dataset_dir = "Dataset"
    _, val_loader, class_names = load_dataset(dataset_dir, batch_size=32)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = "model/fake_login_detector.pth"
    
    if not os.path.exists(model_path):
        print(f"Model file not found at {model_path}")
        return

    print("Loading model...")
    model = build_model(num_classes=len(class_names))
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    print("Evaluating model...")
    y_true = []
    y_pred_probs = []
    y_pred_labels = []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            
            _, predicted = torch.max(outputs, 1)
            
            y_pred_probs.extend(probs.cpu().numpy())
            y_true.extend(labels.numpy())
            y_pred_labels.extend(predicted.cpu().numpy())

    y_true = np.array(y_true)
    y_pred_probs = np.array(y_pred_probs)
    y_pred_labels = np.array(y_pred_labels)
    
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred_labels, target_names=class_names))

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred_labels)
    print("Confusion Matrix:")
    print(cm)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')
    print("Saved confusion matrix plot to confusion_matrix.png")

    # ROC-AUC uses probability estimates of the positive class
    # Phishing is class 0, Real is class 1 (alphabetical)
    auc_score = roc_auc_score(y_true, y_pred_probs[:, 1])
    f1 = f1_score(y_true, y_pred_labels, average='weighted')
    
    print(f"\nFinal ROC-AUC Score: {auc_score:.4f}")
    print(f"Final F1-Score: {f1:.4f}")
    
    # Provide a simple check against user's baseline
    print("\nComparison with Baseline:")
    print("Baseline AUC: 0.748")
    print(f"Current AUC: {auc_score:.4f}")
    if auc_score > 0.748:
        print("Success! The new model outperforms the baseline.")
    else:
        print("The new model did not outperform the baseline.")

if __name__ == "__main__":
    evaluate()

if __name__ == "__main__":
    evaluate()
