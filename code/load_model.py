import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import timm
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = timm.create_model("swin_base_patch4_window7_224", pretrained=False, num_classes=2)
model.load_state_dict(torch.load("model/swin_ai_detector.pth", map_location=device))
model = model.to(device)
model.eval()

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

image_path = "path/to/your/image.jpg" # Change this to your test image path

rgb_img = Image.open(image_path).convert("RGB")
    
input_tensor = test_transform(rgb_img).unsqueeze(0).to(device)

with torch.no_grad():
    output = model(input_tensor)
    probabilities = torch.softmax(output, dim=1)[0]
    
class_names = ["Original (Real)", "AI Generated"]
pred_class_id = torch.argmax(probabilities).item()
pred_class_name = class_names[pred_class_id]
pred_confidence = probabilities[pred_class_id].item() * 100

target_layers = [model.layers[-1].blocks[-1].norm1]

def reshape_transform(tensor, height=7, width=7):
    if len(tensor.shape) == 4:
        B, H, W, C = tensor.shape
        result = tensor.permute(0, 3, 1, 2)
        return result
    elif len(tensor.shape) == 3:
        B, N, C = tensor.shape
        result = tensor.reshape(B, height, width, C)
        result = result.permute(0, 3, 1, 2)
        return result
    return tensor

cam = GradCAM(model=model, target_layers=target_layers, reshape_transform=reshape_transform)
grayscale_cam = cam(input_tensor=input_tensor, targets=None)[0, :]

img_resized = rgb_img.resize((224, 224))
img_array = np.float32(img_resized) / 255.0
visualization = show_cam_on_image(img_array, grayscale_cam, use_rgb=True)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

ax1.imshow(img_resized)
ax1.set_title("Input Test Image", fontsize=12, fontweight='bold')
ax1.axis("off")

ax2.imshow(visualization)
ax2.set_title("Swin Patch Attention (Grad-CAM)", fontsize=12, fontweight='bold')
ax2.axis("off")

plt.suptitle(f"Prediction: {pred_class_name} ({pred_confidence:.2f}% Confidence)", 
             fontsize=14, fontweight='bold', y=0.98, color='darkgreen' if pred_class_id == 0 else 'darkred')

print("--- Prediction Results ---")
for i, name in enumerate(class_names):
    print(f"{name}: {probabilities[i].item()*100:.2f}%")

plt.tight_layout()
plt.show()