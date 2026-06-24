import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms

model_path = r'model\Hornet_M1.pth'
image_path = r'Images\Screenshot 2026-05-20 155120.png'
label_map = {0: "AI-generated", 1: "Real"}

device = torch.device("cpu")
print(f"Using Device: {device}")


print(f"Loading model from {model_path}...")
# map_location='cpu' prevents crash if weights were extracted from a GPU run
loaded_model = torch.load(model_path, map_location=device, weights_only=False)
loaded_model = loaded_model.to(device)
loaded_model.eval()

# Load image
orig_img = Image.open(image_path).convert('RGB')

inference_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

img_tensor = inference_transforms(orig_img).unsqueeze(0).to(device)

with torch.no_grad():
    outputs = loaded_model(img_tensor)
    probabilities = torch.softmax(outputs, dim=1).cpu().numpy()[0]
    predicted_class = np.argmax(probabilities)

pred_label = label_map[predicted_class]
confidence = probabilities[predicted_class] * 100

print("\n" + "="*40)
print("         CLASS PROBABILITIES")
print("="*40)
for class_idx, label_name in label_map.items():
    prob_percent = probabilities[class_idx] * 100
    print(f"-> {label_name:<15}: {prob_percent:.2f}%")
print("="*40)
print(f"Final Prediction: {pred_label} ({confidence:.2f}% Confidence)\n")

gray_img = np.array(orig_img.convert('L'))
h, w = gray_img.shape

fft = np.fft.fft2(gray_img)
fft_shift = np.fft.fftshift(fft)
magnitude_spectrum = 20 * np.log(np.abs(fft_shift) + 1)

y, x = np.indices((h, w))
center_y, center_x = h // 2, w // 2
angles = np.arctan2(y - center_y, x - center_x)
angle_bins = np.linspace(-np.pi, np.pi, 360)
angular_energy = np.zeros(359)

for i in range(359):
    mask = (angles >= angle_bins[i]) & (angles < angle_bins[i+1])
    angular_energy[i] = np.mean(magnitude_spectrum[mask])

# ==========================================
plt.style.use('dark_background')
fig = plt.figure(figsize=(16, 12))

# --- Plot 1: Original Image & Prediction ---
ax1 = plt.subplot(2, 2, 1)
ax1.imshow(orig_img)
ax1.axis('off')
ax1.set_title(f"Prediction: {pred_label}\nConfidence: {confidence:.2f}%", 
              fontsize=14, color='#00ffcc', weight='bold', pad=10)

ax2 = plt.subplot(2, 2, 2)
colors = ('r', 'g', 'b')
img_np = np.array(orig_img)
for i, color in enumerate(colors):
    histogram, bin_edges = np.histogram(img_np[:, :, i], bins=256, range=(0, 256))
    ax2.plot(bin_edges[0:-1], histogram, color=color, alpha=0.8, linewidth=1.5, label=color.upper())

ax2.fill_between(bin_edges[0:-1], histogram, color='gray', alpha=0.1)
ax2.set_title("Pixel-Level Color Frequency", fontsize=12, weight='bold', pad=10)
ax2.set_xlabel("Pixel Intensity Value (0 - 255)", color='#aaaaaa')
ax2.set_ylabel("Pixel Count", color='#aaaaaa')
ax2.grid(True, linestyle='--', alpha=0.3)
ax2.legend()

ax3 = plt.subplot(2, 2, 3)
shading = ax3.imshow(magnitude_spectrum, cmap='magma')
ax3.axis('off')
ax3.set_title("Magnitude Spectrum (FFT)", fontsize=12, weight='bold', pad=10)
fig.colorbar(shading, ax=ax3, fraction=0.046, pad=0.04).ax.tick_params(labelsize=8)

ax4 = plt.subplot(2, 2, 4, projection='polar')
theta = np.linspace(-np.pi, np.pi, 359)
ax4.plot(theta, angular_energy, color='#ffcc00', linewidth=2)
ax4.fill(theta, angular_energy, color='#ffcc00', alpha=0.2)
ax4.set_title("Directional Frequency Energy", fontsize=12, color='#00ffcc', weight='bold', pad=15)
ax4.tick_params(colors='#aaaaaa')
ax4.grid(True, linestyle=':', alpha=0.4)

plt.tight_layout()
plt.show()