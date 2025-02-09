import os
import torch
import random
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont
from dataset import AcneDataset
from model import EfficientNetClassifier
from config import MODEL_SAVE_PATH, NUM_CLASSES

# Set device (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the inference transformation pipeline (resize to 320x320)
inference_transform = transforms.Compose([
    transforms.Resize((320, 320)),  # Use 320x320 as required
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Create an instance of your dataset with the inference transform
data_dir = "./dataset"  # Adjust this path if necessary
dataset = AcneDataset(root_dir=data_dir, transform=inference_transform)

# Randomly select 5 sample indices
sample_indices = random.sample(range(len(dataset)), 5)
images = []       # For transformed images for model input
file_names = []   # Corresponding file paths

for idx in sample_indices:
    image, _ = dataset[idx]
    images.append(image)
    file_names.append(dataset.image_files[idx])  # Assuming your dataset stores file paths

# Stack images into a batch tensor
batch = torch.stack(images).to(device)

# Instantiate and load your pretrained model (EfficientNetClassifier)
model = EfficientNetClassifier(num_classes=NUM_CLASSES)
checkpoint = torch.load(MODEL_SAVE_PATH, map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])
model.to(device)
model.eval()

# Perform inference without tracking gradients
with torch.no_grad():
    outputs = model(batch)
    # Apply softmax to obtain class probabilities
    probabilities = torch.softmax(outputs, dim=1)
    # For each image, get the predicted class index and its confidence (max probability)
    confidences, predictions = torch.max(probabilities, dim=1)

# Mapping from class index to human-readable label
class_mapping = {
    0: "Severe Level 1",
    1: "Severe Level 2",
    2: "Severe Level 3",
    3: "Severe Level 4"
}

# Create output directory for saving annotated images
output_dir = "inference_results"
os.makedirs(output_dir, exist_ok=True)

# For each sample, open the original image (without transform), resize to 320x320, annotate it, and save it
for file_path, pred, conf in zip(file_names, predictions, confidences):
    # Open the original image and resize it to 320x320
    original_image = Image.open(file_path).convert("RGB")
    original_image = original_image.resize((320, 320))
    
    draw = ImageDraw.Draw(original_image)
    
    # Prepare annotation text with mapped label and confidence percentage
    label_text = class_mapping.get(pred.item(), f"Class {pred.item()}")
    text = f"{label_text} ({conf.item()*100:.2f}%)"
    
    # Load a TrueType font if available; otherwise, load the default font
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except IOError:
        font = ImageFont.load_default()
    
    # Get text bounding box using textbbox (fallback to textsize if not available)
    try:
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
    except AttributeError:
        text_width, text_height = draw.textsize(text, font=font)
    
    # Draw a rectangle as background for the text for better visibility
    padding = 4
    rect_coords = [10, 10, 10 + text_width + padding, 10 + text_height + padding]
    draw.rectangle(rect_coords, fill="black")
    
    # Draw the text in white over the rectangle
    draw.text((12, 12), text, fill="white", font=font)
    
    # Construct the output file name and save the annotated image
    base_name = os.path.basename(file_path)
    output_path = os.path.join(output_dir, f"annotated_{base_name}")
    original_image.save(output_path)
    print(f"Saved annotated image to: {output_path}")

print("Inference complete. Check the 'inference_results' directory for annotated images.")
