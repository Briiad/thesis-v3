import os
import sys
import random
import torch
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont

# Debug: print command-line arguments
print("sys.argv:", sys.argv)

# Import model creation functions from your modules
from model.proposed_model import create_mobilenetv3_fcos
from ml.model import EfficientNetClassifier

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------- Transformation Pipelines ----------
detection_transform = transforms.Compose([
    transforms.Resize((640, 640)),
    transforms.ToTensor()
])
classification_transform = transforms.Compose([
    transforms.Resize((320, 320)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ---------- Load Models ----------
detection_model = create_mobilenetv3_fcos(num_classes=7)
detection_checkpoint = torch.load("./best_model.pth", map_location=device)
if "model_state_dict" in detection_checkpoint:
    detection_model.load_state_dict(detection_checkpoint["model_state_dict"])
else:
    detection_model.load_state_dict(detection_checkpoint)
detection_model.to(device)
detection_model.eval()

classification_model = EfficientNetClassifier(num_classes=4)
classif_checkpoint = torch.load("./ml/saved_models/efficientnet.pth", map_location=device)
classification_model.load_state_dict(classif_checkpoint["model_state_dict"])
classification_model.to(device)
classification_model.eval()

# ---------- Mappings ----------
detection_categories = [
    "blackheads", "dark spot", "nodules", 
    "papules", "pustules", "whiteheads"
]
classification_mapping = {
    0: "Severe Level 1",
    1: "Severe Level 2",
    2: "Severe Level 3",
    3: "Severe Level 4"
}

# ---------- Determine Input Image(s) ----------
if len(sys.argv) >= 2:
    # Use the provided image path
    image_paths = [sys.argv[1]]
    print("Using provided image:", sys.argv[1])
else:
    # Otherwise, select 5 random images from the dataset
    dataset_dir = "./ml/dataset"
    all_files = [os.path.join(dataset_dir, f) for f in os.listdir(dataset_dir)
                 if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not all_files:
        print("No images found in", dataset_dir)
        sys.exit(1)
    image_paths = random.sample(all_files, min(5, len(all_files)))
    print("No image provided. Using random images from dataset.")

os.makedirs("./result", exist_ok=True)
try:
    font = ImageFont.truetype("arial.ttf", 16)
except IOError:
    font = ImageFont.load_default()

# ---------- Inference Loop ----------
for image_path in image_paths:
    original_image = Image.open(image_path).convert("RGB")
    
    # Whole-image classification for severity (resize to 320x320)
    classification_image = original_image.resize((320, 320))
    cls_input_tensor = classification_transform(classification_image).unsqueeze(0).to(device)
    with torch.no_grad():
        cls_outputs = classification_model(cls_input_tensor)
        cls_probs = torch.softmax(cls_outputs, dim=1)
        cls_conf, cls_pred = torch.max(cls_probs, dim=1)
    cls_conf_val = cls_conf.item()
    severity_label = classification_mapping.get(cls_pred.item(), f"Class {cls_pred.item()}")
    
    # Detection for bounding boxes (resize to 640x640)
    detection_image = original_image.resize((640, 640))
    det_input_tensor = detection_transform(detection_image).to(device)
    detection_input = [det_input_tensor]
    with torch.no_grad():
        detection_outputs = detection_model(detection_input)
    detection_result = detection_outputs[0]
    boxes = detection_result["boxes"]
    scores = detection_result["scores"]
    det_labels = detection_result.get("labels", None)
    
    annotated_image = detection_image.copy()
    draw = ImageDraw.Draw(annotated_image)
    score_threshold = 0.2
    for idx, (box, score) in enumerate(zip(boxes, scores)):
        if score.item() < score_threshold:
            continue
        x1, y1, x2, y2 = map(int, box.tolist())
        if det_labels is not None:
            label_idx = int(det_labels[idx].item())
            if label_idx < 1 or label_idx > len(detection_categories):
                continue
            detection_label = detection_categories[label_idx - 1]
        else:
            detection_label = "N/A"
        det_conf = score.item()
        draw.rectangle([x1, y1, x2, y2], outline="green", width=2)
        det_text = f"{detection_label} ({det_conf*100:.1f}%)"
        try:
            text_bbox = draw.textbbox((x1, y1), det_text, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
        except AttributeError:
            text_width, text_height = draw.textsize(det_text, font=font)
        padding = 4
        rect_coords = [x1, y1, x1 + text_width + 2*padding, y1 + text_height + 2*padding]
        draw.rectangle(rect_coords, fill="green")
        draw.text((x1 + padding, y1 + padding), det_text, fill="white", font=font)
    
    # Annotate severity in bottom right
    severity_text = f"Severity: {severity_label} ({cls_conf_val*100:.1f}%)"
    try:
        text_bbox = draw.textbbox((0, 0), severity_text, font=font)
        txt_w = text_bbox[2] - text_bbox[0]
        txt_h = text_bbox[3] - text_bbox[1]
    except AttributeError:
        txt_w, txt_h = draw.textsize(severity_text, font=font)
    img_w, img_h = annotated_image.size
    padding = 10
    text_x = img_w - txt_w - padding
    text_y = img_h - txt_h - padding
    bg_rect = [text_x - padding, text_y - padding, img_w, img_h]
    draw.rectangle(bg_rect, fill="green")
    draw.text((text_x, text_y), severity_text, fill="white", font=font)
    
    result_path = os.path.join("./result", f"result_{os.path.basename(image_path)}")
    annotated_image.save(result_path)
    print(f"Saved annotated image to: {result_path}")

print("Inference complete. Check the './result' folder for annotated images.")
