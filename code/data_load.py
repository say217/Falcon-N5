!pip install -q datasets huggingface_hub
from datasets import load_dataset
import matplotlib.pyplot as plt

print("Loading dataset from Hugging Face (ignoring split metadata checks)...")
# Added verification_mode="no_checks" to bypass the metadata size mismatch
dataset = load_dataset("Parveshiiii/AI-vs-Real", verification_mode="no_checks")
print("Dataset loaded successfully!\n")

# 2. Print basic dataset structure
print("## Dataset Structure ##")
print(dataset)
print("-" * 50)

# 3. Access a sample row
sample = dataset["train"][0]
label_map = {0: "AI-generated", 1: "Real"}

print("## Sample Row Metadata ##")
print(f"Image Object: {sample['image']}")
print(f"Label ID: {sample['binary_label']} -> ({label_map[sample['binary_label']]})")
print("-" * 50)

# 4. Display the sample image directly in the notebook
print("Visualizing sample image:")
display(sample['image'])