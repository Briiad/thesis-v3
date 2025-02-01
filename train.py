from multiprocessing import freeze_support

from model.mobilenetv2 import create_ssd_model
from model.mobilenetv3_large import create_mobilenetv3_large_ssd
from data.loader import loaders
from config.config import train_cfg, data_cfg
from utils.trainer import Trainer

if __name__ == '__main__':
  freeze_support()
  # Create model and get dataloaders
  model = create_mobilenetv3_large_ssd(num_classes=data_cfg.num_classes)
  train_loader, val_loader, test_loader = loaders()

  # Initialize trainer
  trainer = Trainer(model, train_loader, val_loader, test_loader, train_cfg)

  # Start training
  final_metrics = trainer.train()