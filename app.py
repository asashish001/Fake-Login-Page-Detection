import os
import torch
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F
from flask import Flask, request, jsonify, render_template

from train import build_model  # Import model architecture to load weights

app = Flask(__name__)

# Constants
MODEL_PATH = "model/fake_login_detector.pth"
CLASS_NAMES = ['Phishing', 'Real'] # Must match training data folder names sorting
IMG_SIZE = (224, 224)

# Load the model globally lazily
model = None
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_model_if_exists():
    global model
    if os.path.exists(MODEL_PATH) and model is None:
        try:
            model = build_model(num_classes=2)
            model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
            model.to(device)
            model.eval()
            print("PyTorch model loaded successfully.")
        except Exception as e:
            print(f"Failed to load model: {e}")

def preprocess_image(image):
    """
    Ensure the image is in RGB format, resizes it, and returns a PyTorch tensor batch.
    """
    if image.mode != 'RGB':
        image = image.convert('RGB')
    transform = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    img_tensor = transform(image)
    img_tensor = img_tensor.unsqueeze(0) # Create a batch
    return img_tensor.to(device)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    load_model_if_exists()
    
    if model is None:
        return jsonify({
            'error': 'Model is not trained or loaded yet. Please wait for the training process to finish and try again.'
        }), 503

    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request (Make sure to POST with "file" key in form-data).'}), 400
        
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected for uploading'}), 400

    if file:
        try:
            # Read image
            image = Image.open(file.stream)
            processed_img = preprocess_image(image)
            
            # Predict
            with torch.no_grad():
                outputs = model(processed_img)
                # Convert logits to probabilities
                probs = F.softmax(outputs, dim=1)
            
            raw_phishing_prob = float(probs[0][0].cpu())
            raw_real_prob = float(probs[0][1].cpu())
            
            # Since the model was trained with a 2.0x penalty factor for the 'Phishing' class
            # it aggressively over-predicts Phishing to minimize False Negatives. 
            # We explicitly counteract this weight here for a more balanced UI display if desired,
            # but to mimic strict security checking we leave it raw or adjust slightly.
            adjusted_phish = raw_phishing_prob / 2.0
            adjusted_real = raw_real_prob
            
            total = adjusted_phish + adjusted_real
            phishing_prob = adjusted_phish / total
            real_prob = adjusted_real / total
            
            if phishing_prob > real_prob:
                predicted_class = CLASS_NAMES[0]
                confidence = phishing_prob * 100
            else:
                predicted_class = CLASS_NAMES[1]
                confidence = real_prob * 100
                
            return jsonify({
                'prediction': predicted_class,
                'confidence': f"{confidence:.2f}%",
                'phishing_prob': phishing_prob,
                'real_prob': real_prob
            })
            
        except Exception as e:
            return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Ensure templates and static folders exist
    os.makedirs('templates', exist_ok=True)
    os.makedirs('static/css', exist_ok=True)
    os.makedirs('static/js', exist_ok=True)
    
    # Try loading the model initially
    load_model_if_exists()
    
    app.run(debug=True, port=5000)
