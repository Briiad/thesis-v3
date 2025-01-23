import os
import torch
import numpy as np
from PIL import Image
from pycocotools.coco import COCO
import albumentations as A
from albumentations.pytorch import ToTensorV2

class COCODataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, ann_file, transform=None, target_transform=None):
        """
        Args:
            root_dir (str): Directory with all the images
            ann_file (str): Path to COCO annotation file
            transform (callable, optional): Optional transform to be applied on images
            target_transform (callable, optional): Optional transform to be applied on annotations
        """
        self.root_dir = root_dir
        self.coco = COCO(ann_file)
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is a dictionary containing COCO format annotations
        """
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        annotations = coco.loadAnns(ann_ids)

        # Load image
        img_info = coco.loadImgs(img_id)[0]
        image_path = os.path.join(self.root_dir, img_info['file_name'])
        image = np.array(Image.open(image_path).convert('RGB'))

        # Convert annotations to the format needed
        boxes = []
        labels = []
        masks = []
        mask_tensors = []
        
        for ann in annotations:
            # Get bbox
            x, y, w, h = ann['bbox']
            boxes.append([x, y, x + w, y + h])
            labels.append(ann['category_id'])
            
            # Get segmentation mask if available
            if 'segmentation' in ann and ann['segmentation']:
              if isinstance(ann['segmentation'], list):
                  # Polygon format - only create mask if polygon exists
                  if ann['segmentation'][0]:
                      mask = coco.annToMask(ann)
                      mask_tensor = torch.from_numpy(mask).to(dtype=torch.uint8)
                      mask_tensors.append(mask_tensor)
              elif isinstance(ann['segmentation'], dict):
                  # RLE format
                  mask_tensor = torch.from_numpy(mask).to(dtype=torch.uint8)
                  mask_tensors.append(mask_tensor)

        # Convert to numpy arrays
        boxes = np.array(boxes, dtype=np.float32)
        labels = np.array(labels, dtype=np.int64)
        masks = torch.stack(mask_tensors) if mask_tensors else torch.zeros(
            (0, img_info['height'], img_info['width']), 
            dtype=torch.uint8
        )

        target = {
            'boxes': boxes,
            'labels': labels,
            'masks': masks,
            'image_id': torch.tensor([img_id])
        }

        # Apply transforms
        if self.transform is not None:
            transformed = self.transform(
                image=image,
                bboxes=boxes,
                labels=labels,
                masks=masks.numpy() if masks.numel() > 0 else None
            )
            
            image = transformed['image']
            target['boxes'] = torch.as_tensor(transformed['bboxes'], dtype=torch.float32)
            target['labels'] = torch.as_tensor(transformed['labels'], dtype=torch.int64)
            # Fix mask handling
            if masks.numel() > 0 and 'masks' in transformed:
                target['masks'] = torch.as_tensor(transformed['masks'], dtype=torch.uint8)
            else:
                target['masks'] = torch.zeros((0, image.shape[1], image.shape[2]), dtype=torch.uint8)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return image, target

    def __len__(self):
        return len(self.ids)