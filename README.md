# Visual Phishing Detector - Deep Learning & Computer Vision

This project leverages **Deep Learning**, **Computer Vision**, and **Optical Character Recognition (OCR)** to detect phishing websites by analyzing screenshots of their login pages. Modern phishing attacks often perfectly replicate the visual styling of legitimate sites to bypass traditional detectors. This solution uses a hybrid approach: a **ResNet50** Convolutional Neural Network (CNN) built with **PyTorch** to analyze global visual "DNA", combined with an **EasyOCR-powered engine** that extracts the URL from the browser's address bar to detect unauthorized brand domain usage (e.g., a Facebook lookalike page hosted on a foreign domain).

## ✨ Key Features

- **Hybrid Visual-OCR Engine**: Combines ResNet50 visual pattern matching with OCR text scanning.
- **Automated Address Bar Extraction**: Crops the top 12% of screenshots to locate the browser address bar, extracts the URL, and parses the root domain.
- **Brand Domain Spoofing Verification**: Cross-references recognized brands (Facebook, Google, PayPal, Netflix, Microsoft, etc.) with official legitimate domain suffix lists, overriding predictions if a mismatch is found.
- **ResNet50 Backbone**: Utilizes the powerful ResNet50 architecture (pre-trained on ImageNet) for robust spatial visual feature extraction.
- **PyTorch Migration**: Fully implemented in PyTorch for high performance and flexible model training.
- **Aggressive False Negative Penalization**: Implements custom class weights with a **2.0x penalty factor** for phishing samples to ensure high recall in security contexts.
- **Modern Dashboard**: A premium, dark-mode glassmorphic UI built with Tailwind CSS for real-time interaction and prediction, now featuring live brand discrepancy reasoning.
- **Flask REST API**: Model predictions are served via an asynchronous Flask backend.

## 🛠️ Tech Stack

- **Backend**: Python, PyTorch, Torchvision, Flask, EasyOCR
- **Frontend**: HTML5, Vanilla JavaScript, Tailwind CSS (Glassmorphism), Vue.js
- **Data Science**: Scikit-learn, Numpy, Matplotlib, Seaborn, PIL, OpenCV

## 📂 Project Structure

```text
LPD/
├── Dataset/                     # Image dataset (requires structure below)
│   ├── phishing/               # Screenshots of fake login pages
│   └── real/                   # Screenshots of legitimate login pages
├── static/                      # Static assets for the Web UI
│   ├── css/style.css           # Glassmorphism and custom styles
│   └── js/main.js              # Async upload & UI interaction logic
├── templates/
│   └── index.html              # Modern web dashboard
├── model/
│   └── fake_login_detector.pth # Saved PyTorch model weights
├── app.py                      # Flask server & Prediction API
├── dataset_loader.py           # PyTorch DataLoader & Augmentation pipeline
├── train.py                    # Script to train and fine-tune ResNet50
├── evaluate.py                 # Generates ROC-AUC, F1-Score, and Confusion Matrix
└── README.md                   # Project documentation
```

## 🚀 Getting Started

### 1. Prerequisites
Ensure you have Python 3.8+ installed. Install the required dependencies using the provided `requirements.txt` file:

```bash
pip install -r requirements.txt
```
*(Note: The requirements file installs the CPU version of PyTorch by default. Modify it or install PyTorch manually if you need CUDA support)*.

### 2. Dataset Setup
Populate the `Dataset/phishing` and `Dataset/real` folders with `.png` or `.jpg` screenshots.

### 3. Model Training
Train the ResNet50 model with transfer learning:
```bash
python train.py
```
> The script is configured to freeze early layers and fine-tune the final convolution block (`layer4`) for optimal speed/accuracy trade-offs.

### 4. Model Evaluation
Assess performance and compare against the baseline:
```bash
python evaluate.py
```
This generates `confusion_matrix.png` and provides detailed metrics including **ROC-AUC** and **F1-Score**.

### 5. Running the Web App
Launch the interactive dashboard:
```bash
python app.py
```
Access the UI at `http://127.0.0.1:5000`. Drag and drop any screenshot to verify its authenticity.

---
**📊 Performance Note**: This model is specifically tuned to surpass the baseline AUC of **0.748** by focusing on minimizing False Negatives through weighted cross-entropy loss.
