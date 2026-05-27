import os
import torch
from torchvision import transforms
from PIL import Image, ImageEnhance
import torch.nn.functional as F
from flask import Flask, request, jsonify, render_template
import easyocr
import re
import numpy as np
from urllib.parse import urlparse

from train import build_model  # Import model architecture to load weights

app = Flask(__name__)

# Constants
MODEL_PATH = "model/fake_login_detector.pth"
CLASS_NAMES = ['Phishing', 'Real'] # Must match training data folder names sorting
IMG_SIZE = (224, 224)

# Load the model globally lazily
model = None
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize EasyOCR reader globally
print("Initializing EasyOCR reader...")
reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available(), verbose=False)

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

def select_best_url(candidates):
    """
    Selects the most legitimate public URL candidate from a list of OCR-detected strings,
    filtering out local addresses (localhost, 127.0.0.1, 192.168.*), version numbers, or short numeric patterns.
    """
    if not candidates:
        return None
        
    valid_candidates = []
    
    for cand in candidates:
        cand_lower = cand.lower()
        
        # 1. Skip local development hosts, loopback IPs, or default ports
        if '127.0.0.1' in cand_lower or 'localhost' in cand_lower or '192.168.' in cand_lower or '0.0.0.0' in cand_lower:
            continue
            
        # 2. Skip numeric version numbers, dates, or file sizes (e.g. "3.10.6", "2.10.0")
        non_num_chars = re.sub(r'[0-9.-]', '', cand_lower)
        if len(non_num_chars) < 2:  # very few letters, likely a version or date
            continue
            
        valid_candidates.append(cand)
        
    # If we have public candidates, sort by alphabetical character density to prioritize real domains
    if valid_candidates:
        valid_candidates.sort(key=lambda x: (len(re.sub(r'[^a-zA-Z]', '', x)), len(x)), reverse=True)
        return valid_candidates[0]
        
    # Return None if no valid candidate exists, skipping mismatch false positives
    return None

def extract_url_from_image(image):
    """
    Highly optimized, space-tolerant, and typo-resilient URL/domain extractor using EasyOCR.
    Utilizes original high-resolution cropping and adaptive upscaling/enhancement for tiny address bar text.
    """
    width, height = image.size
    
    # Create speed-efficient downscaled image for fallback full-page scans
    target_width = 1000
    if width > target_width:
        scale_res = target_width / width
        image_resized = image.resize((target_width, int(height * scale_res)), Image.Resampling.LANCZOS)
    else:
        image_resized = image
        
    # 1. Crop address bar from the ORIGINAL high-resolution image to preserve tiny details!
    top_crop_height = int(height * 0.18)
    top_crop = image.crop((0, 0, width, top_crop_height))
    
    crop_w, crop_h = top_crop.size
    
    # 2. Adaptive upscaling to expand small characters
    # If the cropped address bar height is small (e.g. < 120px), upscale by 2.5x to expand characters
    if crop_h < 120:
        scale = 2.5
    elif crop_h > 240:
        # If the crop is very large, downscale to a standardized height for speed
        scale = 180.0 / crop_h
    else:
        scale = 1.0
        
    if scale != 1.0:
        top_crop = top_crop.resize((int(crop_w * scale), int(crop_h * scale)), Image.Resampling.LANCZOS)
        
    # 3. Apply grayscale and contrast enhancements to clarify character borders
    processed_crop = top_crop.convert('L')
    enhancer = ImageEnhance.Contrast(processed_crop)
    processed_crop = enhancer.enhance(1.8)
    
    # 4. Perform fast OCR on preprocessed crop
    img_np = np.array(processed_crop)
    results = reader.readtext(img_np, detail=0)
    top_text = " ".join(results)
    
    print(f"\n[DEBUG OCR] Raw text detected: {results}", flush=True)
    
    # 4. advanced clean of spacing and punctuation errors typical in browser URLs
    # A. Remove spaces around hyphens, dots, slashes, and colons
    cleaned_top_text = re.sub(r'\s*-\s*', '-', top_text)
    cleaned_top_text = re.sub(r'\s*\.\s*', '.', cleaned_top_text)
    cleaned_top_text = re.sub(r'\s*/\s*', '/', cleaned_top_text)
    cleaned_top_text = re.sub(r'\s*:\s*', ':', cleaned_top_text)
    
    # B. Replace commas between alphanumeric characters with dots (e.g., "my,id" -> "my.id")
    cleaned_top_text = re.sub(r'([a-zA-Z0-9-]+),([a-zA-Z0-9-]+)', r'\1.\2', cleaned_top_text)
    
    # C. Replace spaces preceding common domain suffixes with dots (e.g., "my id" -> "my.id")
    cleaned_top_text = re.sub(r'\s+(com|net|org|id|co|info|xyz|club|gov|edu|my)\b', r'.\1', cleaned_top_text)
    cleaned_top_text = re.sub(r'\s*\.\s*', '.', cleaned_top_text) # Re-clean any newly formed dots
    
    # D. Correct common OCR letter-to-number substitutions in suffixes
    cleaned_top_text = re.sub(r'\.c[0oO]m\b', '.com', cleaned_top_text)
    cleaned_top_text = re.sub(r'\.n[3e]t\b', '.net', cleaned_top_text)
    cleaned_top_text = re.sub(r'\.[0oO]rg\b', '.org', cleaned_top_text)
    cleaned_top_text = re.sub(r'\.[1lI]d\b', '.id', cleaned_top_text)
    
    # E. Reconstruct concatenated domain extensions (e.g., "facebookcom" -> "facebook.com")
    cleaned_top_text = re.sub(r'\b([a-zA-Z0-9-]+)(com|net|org|co|id|xyz|info|club|gov|edu|uk)\b', r'\1.\2', cleaned_top_text)
    
    print(f"[DEBUG OCR] Cleaned text: {cleaned_top_text}", flush=True)
    
    # Lenient regex to match domains with alphabetical TLDs only (no digits)
    url_pattern = re.compile(
        r'([a-zA-Z0-9-]{2,}\.)+[a-zA-Z]{2,8}(/[a-zA-Z0-9-._~:/?#\[\]@!$&\'()*+,;=]*)?'
    )
    
    matches = url_pattern.finditer(cleaned_top_text)
    candidates = [m.group(0) for m in matches]
    extracted_url = select_best_url(candidates)
    
    # 5. Full page scan fallback if address bar not caught at top
    if not extracted_url:
        print("[DEBUG OCR] URL not found in top crop. Performing full page fallback scan...", flush=True)
        full_img_np = np.array(image_resized)
        full_results = reader.readtext(full_img_np, detail=0)
        full_text = " ".join(full_results)
        
        cleaned_full_text = re.sub(r'\s*-\s*', '-', full_text)
        cleaned_full_text = re.sub(r'\s*\.\s*', '.', cleaned_full_text)
        cleaned_full_text = re.sub(r'\s*/\s*', '/', cleaned_full_text)
        cleaned_full_text = re.sub(r'\s*:\s*', ':', cleaned_full_text)
        cleaned_full_text = re.sub(r'([a-zA-Z0-9-]+),([a-zA-Z0-9-]+)', r'\1.\2', cleaned_full_text)
        cleaned_full_text = re.sub(r'\s+(com|net|org|id|co|info|xyz|club|gov|edu|my)\b', r'.\1', cleaned_full_text)
        cleaned_full_text = re.sub(r'\s*\.\s*', '.', cleaned_full_text)
        
        # Apply TLD and concatenation cleaning in fallback too
        cleaned_full_text = re.sub(r'\.c[0oO]m\b', '.com', cleaned_full_text)
        cleaned_full_text = re.sub(r'\.n[3e]t\b', '.net', cleaned_full_text)
        cleaned_full_text = re.sub(r'\.[0oO]rg\b', '.org', cleaned_full_text)
        cleaned_full_text = re.sub(r'\.[1lI]d\b', '.id', cleaned_full_text)
        cleaned_full_text = re.sub(r'\b([a-zA-Z0-9-]+)(com|net|org|co|id|xyz|info|club|gov|edu|uk)\b', r'\1.\2', cleaned_full_text)
        
        full_matches = url_pattern.finditer(cleaned_full_text)
        full_candidates = [m.group(0) for m in full_matches]
        extracted_url = select_best_url(full_candidates)
        page_text = full_text
    else:
        # Get full text for brand context
        full_img_np = np.array(image_resized)
        full_results = reader.readtext(full_img_np, detail=0)
        page_text = " ".join(full_results)
        
    if extracted_url:
        print(f"[DEBUG OCR] MATCHED URL SUCCESS: {extracted_url}\n", flush=True)
        return extracted_url, page_text
        
    print("[DEBUG OCR] MATCHED URL FAILED\n", flush=True)
    return None, page_text

def get_domain_from_url(url_str):
    """
    Parses and extracts the clean registered root domain/host from an extracted URL string.
    """
    if not url_str.startswith(('http://', 'https://')):
        url_str = 'http://' + url_str
    try:
        parsed = urlparse(url_str)
        netloc = parsed.netloc.lower()
        if ":" in netloc:
            netloc = netloc.split(":")[0]
        return netloc
    except Exception:
        return None

# Dictionary of high-profile target brands and their official legitimate domains
BRAND_DOMAINS = {
    'facebook': ['facebook.com', 'facebook.co', 'fb.com', 'messenger.com', 'instagram.com'],
    'google': ['google.com', 'gmail.com', 'youtube.com', 'google.co', 'googleapis.com'],
    'microsoft': ['microsoft.com', 'live.com', 'outlook.com', 'office.com', 'office365.com', 'msn.com', 'bing.com'],
    'paypal': ['paypal.com', 'paypal.co'],
    'netflix': ['netflix.com'],
    'apple': ['apple.com', 'icloud.com'],
    'steam': ['steampowered.com', 'steamcommunity.com'],
    'dropbox': ['dropbox.com', 'dropboxstatic.com'],
    'amazon': ['amazon.com', 'amazon.co'],
    'yahoo': ['yahoo.com']
}

def analyze_domain_brand_mismatch(domain, page_text, extracted_url):
    """
    Checks if a well-known brand is visually/textually present, but the domain 
    is NOT affiliated with that brand.
    """
    if not domain:
        return False, ""
        
    combined_text = (page_text + " " + (extracted_url or "")).lower()
    
    for brand, legit_domains in BRAND_DOMAINS.items():
        # Match brand name as a word
        brand_pattern = re.compile(rf'\b{brand}\b')
        if brand_pattern.search(combined_text):
            # Check if domain matches any legitimate suffix
            is_legit = False
            for legit in legit_domains:
                if domain == legit or domain.endswith('.' + legit):
                    is_legit = True
                    break
            
            if not is_legit:
                return True, (
                    f"Brand discrepancy detected! The page contains elements or text referencing '{brand.capitalize()}', "
                    f"but it is hosted on the unauthorized domain '{domain}' (expected official domain like: {', '.join(legit_domains)})."
                )
                
    return False, ""

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
            
            # OCR URL Extraction & Brand Domain Heuristic Checks
            extracted_url = None
            extracted_domain = None
            ocr_phishing_detected = False
            ocr_reason = ""
            
            try:
                extracted_url, page_text = extract_url_from_image(image)
                if extracted_url:
                    extracted_domain = get_domain_from_url(extracted_url)
                    ocr_phishing_detected, ocr_reason = analyze_domain_brand_mismatch(extracted_domain, page_text, extracted_url)
            except Exception as ocr_err:
                print(f"OCR URL extraction error (continuing with visual classification): {ocr_err}")

            # Visual classification model predict
            processed_img = preprocess_image(image)
            with torch.no_grad():
                outputs = model(processed_img)
                # Convert logits to probabilities
                probs = F.softmax(outputs, dim=1)
            
            raw_phishing_prob = float(probs[0][0].cpu())
            raw_real_prob = float(probs[0][1].cpu())
            
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
            
            # Override prediction if OCR checks caught a brand-domain mismatch
            if ocr_phishing_detected:
                predicted_class = 'Phishing'
                confidence = 100.00
                reason_msg = ocr_reason
            else:
                reason_msg = f"Visual classifier predicted {predicted_class} with {confidence:.2f}% confidence."
                if extracted_url:
                    reason_msg += f" Detected URL: {extracted_url} (Parsed Domain: {extracted_domain})."
                else:
                    reason_msg += " No browser URL detected in the screenshot."

            return jsonify({
                'prediction': predicted_class,
                'confidence': f"{confidence:.2f}%",
                'phishing_prob': phishing_prob,
                'real_prob': real_prob,
                'extracted_url': extracted_url,
                'extracted_domain': extracted_domain,
                'reason': reason_msg
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
