import os
import re
from PIL import Image
from torch.utils.data import Dataset
from collections import Counter

class AcneDataset(Dataset):
  def __init__(self, root_dir, transform=None):
    self.root_dir = root_dir
    self.transform = transform
    self.image_files = [os.path.join(root_dir, f) for f in os.listdir(root_dir)
              if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    # Optionally, print the detailed class distribution here:
    labels = []
    for path in self.image_files:
      match = re.search(r'levle(\d)', os.path.basename(path))
      if match:
        labels.append(int(match.group(1)))
      else:
        raise ValueError(f"Label not found in filename: {os.path.basename(path)}")
    distribution = Counter(labels)
    print("Detailed class distribution in the dataset:")
    total_samples = len(self.image_files)
    for label, count in sorted(distribution.items()):
      percentage = (count / total_samples) * 100 if total_samples > 0 else 0
      print(f"  Label {label}: {count} sample{'s' if count > 1 else ''} ({percentage:.2f}%)")
  
  def __len__(self):
    return len(self.image_files)
  
  def __getitem__(self, idx):
    img_path = self.image_files[idx]
    image = Image.open(img_path).convert("RGB")
    if self.transform:
      image = self.transform(image)
    match = re.search(r'levle(\d)', os.path.basename(img_path))
    if match:
      label = int(match.group(1))
    else:
      raise ValueError(f"Label not found in filename: {os.path.basename(img_path)}")
    return image, label

# For quick testing:
if __name__ == '__main__':
  dataset = AcneDataset(root_dir="./dataset")
