from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

from config.config import data_cfg as data_config
from data.dataset import COCODataset

def get_transform(train=True):
    """
    Returns albumentations transforms pipeline based on config
    Args:
        train (bool): If True, return transforms for training, else for validation
    """
    if train:
        return A.Compose([
            A.RandomResizedCrop(
                size=data_config.img_size
            ),
            A.HorizontalFlip(p=data_config.flip_prob),
            A.RandomBrightnessContrast(p=data_config.brightness_contrast_prob),
            A.Rotate(limit=30, p=data_config.rotate_prob),
            A.Normalize(
                mean=data_config.mean,
                std=data_config.std,
            ),
            ToTensorV2(),
        ], bbox_params=A.BboxParams(
               format='coco',
               label_fields=['labels']
           ), is_check_shapes=False)
    else:
        return A.Compose([
            A.Resize(
                height=data_config.img_size[0],
                width=data_config.img_size[1]
            ),
            A.Normalize(
                mean=data_config.mean,
                std=data_config.std,
            ),
            ToTensorV2(),
        ], bbox_params=A.BboxParams(
               format='coco',
               label_fields=['labels']
           ), is_check_shapes=False)

def get_data_loader(root_dir, ann_file, train=True):
    """
    Returns DataLoader for COCO dataset using config settings
    Args:
        root_dir (str): Directory with all the images
        ann_file (str): Path to COCO annotation file
        train (bool): If True, use training transforms
    """
    transform = get_transform(train=train)
    dataset = COCODataset(root_dir, ann_file, transform=transform)
    
    # Update config with dataset information if not set
    if data_config.categories is None:
        data_config.categories = dataset.coco.loadCats(dataset.coco.getCatIds())
        data_config.num_classes = len(data_config.categories)
    
    return DataLoader(
        dataset,
        batch_size=data_config.batch_size,
        shuffle=train,
        num_workers=data_config.num_workers,
        collate_fn=collate_fn,
        pin_memory=data_config.pin_memory
    )

def collate_fn(batch):
    """Custom collate function for handling variable size images and annotations"""
    return tuple(zip(*batch))

# Example usage with config
def loaders():
    """Helper function to get both train and validation loaders"""
    train_loader = get_data_loader(
        root_dir=data_config.train_root_dir,
        ann_file=data_config.train_ann_file,
        train=True
    )
    
    val_loader = get_data_loader(
        root_dir=data_config.val_root_dir,
        ann_file=data_config.val_ann_file,
        train=False
    )
    
    test_loader = get_data_loader(
        root_dir=data_config.test_root_dir,
        ann_file=data_config.test_ann_file,
        train=False
    )
    
    return train_loader, val_loader, test_loader