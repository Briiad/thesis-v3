from multiprocessing import freeze_support

from model.mobilenetv2_nonbipfn import create_mobilenetv2_ssd
from model.mobilenetv3_large_nonbipfn import create_mobilenetv3_large_ssd
from model.proposed_model import create_mobilenetv3_fcos
from model.efficientnet import get_retinanet_model
from data.loader import train_dataset, val_dataset, test_dataset
from config.config import train_cfg, data_cfg
from utils.trainer import Trainer

if __name__ == '__main__':
  freeze_support()
  # Create model and get dataloaders
  model = create_mobilenetv3_fcos(num_classes=data_cfg.num_classes)

  # Initialize trainer
  trainer = Trainer(model=model, train_loader=train_dataset, val_loader=val_dataset, test_loader=test_dataset, config=train_cfg)

  # Start training
  final_metrics = trainer.train()