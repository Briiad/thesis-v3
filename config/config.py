# config.py
from dataclasses import dataclass, field
from typing import Tuple, List

@dataclass
class DataConfig:
    # Dataset paths
    train_root_dir: str = "dataset/train"
    val_root_dir: str = "dataset/valid"
    test_root_dir: str = "dataset/test"
    train_ann_file: str = "dataset/train/_annotations.coco.json"
    val_ann_file: str = "dataset/valid/_annotations.coco.json"
    test_ann_file: str = "dataset/test/_annotations.coco.json"
    
    # Image settings
    img_size: Tuple[int, int] = (320, 320)
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
    
    # Dataloader settings
    batch_size: int = 16
    num_workers: int = 4
    pin_memory: bool = True
    
    # Augmentation probabilities
    flip_prob: float = 0.5
    brightness_contrast_prob: float = 0.2
    rotate_prob: float = 0.3
    
    # Categories
    categories: List[str] = field(default_factory=lambda: ["blackheads", "dark spot", "nodules", "papules", "pustules", "whiteheads"])
    num_classes: int = 6

@dataclass
class TrainConfig:
    # Training settings
    epochs: int = 10
    learning_rate: float = 1e-3
    weight_decay: float = 0.0005
    lr_scheduler_step: int = 30
    lr_scheduler_gamma: float = 0.1
    
    # Checkpoint settings
    checkpoint_dir: str = "checkpoints"
    save_frequency: int = 5

# Create default config instances
data_cfg = DataConfig()
train_cfg = TrainConfig()