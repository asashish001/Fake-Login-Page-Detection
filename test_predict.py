import torch
from torchvision import transforms
from PIL import Image
from train import build_model
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = build_model(num_classes=2)
model.load_state_dict(torch.load("model/fake_login_detector.pth", map_location=device))
model.to(device)
model.eval()

image = Image.open("Dataset/phishing/picass0.com_227.png").convert('RGB')
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
img_tensor = transform(image).unsqueeze(0).to(device)

with torch.no_grad():
    outputs = model(img_tensor)
    probs = F.softmax(outputs, dim=1)
    
print(f"Predictions (0=Phishing, 1=Real): {probs[0].cpu().numpy()}")
