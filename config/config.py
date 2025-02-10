# config.py
from dataclasses import dataclass, field
from typing import Tuple, List

@dataclass
class DataConfig:
    # Dataset paths
    root_dir: str = "dataset"
    train_dir: str = "dataset/train"
    valid_dir: str = "dataset/valid"
    test_dir: str = "dataset/test"
    
    # Image settings
    img_size: Tuple[int, int] = (640, 640)
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
    categories: List[str] = field(default_factory=lambda: [
        "blackheads", "dark spot", "nodules", 
        "papules", "pustules", "whiteheads"
    ])
    
    # Number of classes (including background)
    num_classes: int = 7  # 6 classes + 1 background

@dataclass
class TrainConfig:
    # Training settings
    epochs: int = 70
    num_classes: int = 7
    learning_rate: float = 0.01
    weight_decay: float = 1e-4
    lr_scheduler_step: int = 25
    lr_scheduler_gamma: float = 0.1
    
    # Checkpoint settings
    checkpoint_dir: str = "outputs/checkpoints"
    save_frequency: int = 5

# Create default config instances
data_cfg = DataConfig()
train_cfg = TrainConfig()