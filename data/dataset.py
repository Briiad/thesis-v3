import os
import xml.etree.ElementTree as ET
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np

class CustomVOCDataset(Dataset):
    def __init__(self, data_dir, img_size=(320, 320), 
                 mean=(0.485, 0.456, 0.406), 
                 std=(0.229, 0.224, 0.225), 
                 categories=None,
                 transform=None,
                 flip_prob=0.5,
                 brightness_contrast_prob=0.2,
                 rotate_prob=0.3,
                 num_classes=7):
        """
        Custom VOC-style Dataset for Object Detection with robust filename handling
        """
        self.data_dir = data_dir
        self.img_size = img_size
        self.num_classes = num_classes
        
        # Get all valid image and annotation pairs
        self.valid_samples = self._get_valid_samples()
        
        # Categories
        self.categories = categories or []
        # Adjust indexing to start at 1 (0 is reserved for background)
        self.cat_to_idx = {cat: i+1 for i, cat in enumerate(self.categories)}
        
        # Create transform
        if transform is None:
            self.transform = A.Compose([
                A.Resize(self.img_size[0], self.img_size[1]),
                A.Normalize(mean=mean, std=std),
                ToTensorV2()
            ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))
        else:
            self.transform = transform
    
    def _get_valid_samples(self):
        """
        Find valid image-annotation pairs, handling complex filenames
        """
        valid_samples = []
        for filename in os.listdir(self.data_dir):
            # Strip any potential newline characters and whitespace
            filename = filename.strip()
            
            # Check for image files (multiple possible extensions)
            if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                base_name = os.path.splitext(filename)[0]
                
                # Try to find corresponding XML
                xml_candidates = [
                    f"{base_name}.xml",
                    f"{base_name}_ann.xml",
                    f"{base_name}.annotation.xml"
                ]
                
                for xml_name in xml_candidates:
                    xml_path = os.path.join(self.data_dir, xml_name)
                    if os.path.exists(xml_path):
                        valid_samples.append({
                            'image': os.path.join(self.data_dir, filename),
                            'annotation': xml_path
                        })
                        break
        
        return valid_samples
    
    def __len__(self):
        return len(self.valid_samples)
    
    def _parse_annotation(self, xml_path):
      """
      Parse XML annotation file and normalize bounding boxes
      """
      try:
          tree = ET.parse(xml_path)
          root = tree.getroot()
          
          # Get image size for normalization
          size = root.find('size')
          img_width = int(float(size.find('width').text))
          img_height = int(float(size.find('height').text))
          
          bboxes = []
          labels = []
          
          for obj in root.findall('.//object'):
              name = obj.find('name').text
              
              # Skip if category not in our list
              if name not in self.categories:
                  continue
              
              bbox = obj.find('bndbox')
              xmin = int(float(bbox.find('xmin').text))
              ymin = int(float(bbox.find('ymin').text))
              xmax = int(float(bbox.find('xmax').text))
              ymax = int(float(bbox.find('ymax').text))
              
              # Normalize coordinates
              norm_xmin = max(0, min(xmin / img_width, 1.0))
              norm_ymin = max(0, min(ymin / img_height, 1.0))
              norm_xmax = max(0, min(xmax / img_width, 1.0))
              norm_ymax = max(0, min(ymax / img_height, 1.0))
              
              bboxes.append([norm_xmin, norm_ymin, norm_xmax, norm_ymax])
              labels.append(self.cat_to_idx[name])
          
          return bboxes, labels
      except Exception as e:
          print(f"Error parsing annotation {xml_path}: {e}")
          return [], []
    
    def __getitem__(self, idx):
        # Get image and annotation paths
        sample = self.valid_samples[idx]
        img_path = sample['image']
        xml_path = sample['annotation']
        
        # Read image with error handling
        try:
            image = cv2.imread(img_path)
            if image is None:
                raise ValueError(f"Could not read image: {img_path}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except Exception as e:
            print(f"Error reading image {img_path}: {e}")
            # Return a dummy sample or raise
            return torch.zeros(3, self.img_size[0], self.img_size[1]), [], []
        
        # Parse annotations
        bboxes, labels = self._parse_annotation(xml_path)
        
        if not bboxes:
            print(f"No bounding boxes found for {img_path}")
        
        # Apply transformations
        transformed = self.transform(
            image=image,
            bboxes=bboxes,
            labels=labels
        )
        
        return transformed['image'], transformed['bboxes'], transformed['labels']