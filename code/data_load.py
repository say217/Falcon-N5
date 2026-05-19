!pip install -q datasets huggingface_hub

from datasets import load_dataset
import matplotlib.pyplot as plt

dataset = load_dataset("Rajarshi-Roy-research/Defactify_Image_Dataset")

print(dataset)

sample = dataset["train"][0]
print(sample.keys())


print("Label_A:", sample["Label_A"])
print("Label_B:", sample["Label_B"])

img = sample["Image"]

plt.imshow(img)
plt.axis("off")
plt.show()

print("Caption:", sample["Caption"])