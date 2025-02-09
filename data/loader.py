import torch
from torch.utils.data import DataLoader
from data.dataset import CustomVOCDataset
from config.config import data_cfg

def collate_fn(batch):
    images = [item[0] for item in batch]
    targets = []
    
    for item in batch:
        if len(item[1]) == 0:
            target = {
                'boxes': torch.zeros((0, 4), dtype=torch.float32),
                'labels': torch.zeros((0,), dtype=torch.long)
            }
        else:
            # Convert normalized coordinates back to absolute pixels
            img_h, img_w = images[0].shape[-2:]
            abs_boxes = torch.tensor(item[1], dtype=torch.float32)
            abs_boxes[:, [0, 2]] *= img_w  # Scale x-coordinates
            abs_boxes[:, [1, 3]] *= img_h  # Scale y-coordinates

            target = {
                'boxes': abs_boxes,
                'labels': torch.tensor(item[2], dtype=torch.long)
            }
        targets.append(target)

    return torch.stack(images), targets

def loaders():
    """
    Create train, validation, and test data loaders
    
    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # Create datasets
    train_dataset = CustomVOCDataset(
        data_dir=data_cfg.train_dir,
        img_size=data_cfg.img_size,
        mean=data_cfg.mean,
        std=data_cfg.std,
        categories=data_cfg.categories,
        flip_prob=data_cfg.flip_prob,
        brightness_contrast_prob=data_cfg.brightness_contrast_prob,
        rotate_prob=data_cfg.rotate_prob
    )
    
    val_dataset = CustomVOCDataset(
        data_dir=data_cfg.valid_dir,
        img_size=data_cfg.img_size,
        mean=data_cfg.mean,
        std=data_cfg.std,
        categories=data_cfg.categories
    )
    
    test_dataset = CustomVOCDataset(
        data_dir=data_cfg.test_dir,
        img_size=data_cfg.img_size,
        mean=data_cfg.mean,
        std=data_cfg.std,
        categories=data_cfg.categories
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=data_cfg.batch_size, 
        shuffle=True,
        num_workers=data_cfg.num_workers,
        pin_memory=data_cfg.pin_memory,
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=data_cfg.batch_size, 
        shuffle=False,
        num_workers=data_cfg.num_workers,
        pin_memory=data_cfg.pin_memory,
        collate_fn=collate_fn
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=data_cfg.batch_size, 
        shuffle=False,
        num_workers=data_cfg.num_workers,
        pin_memory=data_cfg.pin_memory,
        collate_fn=collate_fn
    )
    
    return train_loader, val_loader, test_loader