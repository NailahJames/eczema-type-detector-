import torch
from torchvision import transforms
from PIL import Image

# Eczema class names used during training
classes = ['atopic', 'contact', 'dyshidrotic', 'nummular']

# Load the trained model
model = torch.load("model.pt", map_location=torch.device("cpu"))
model.eval()

# Image transform (same as training)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

def predict(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
    return classes[predicted.item()]
