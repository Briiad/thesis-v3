from multiprocessing import freeze_support

from model.mobilenetv2_nonbipfn import create_mobilenetv2_ssd
from model.mobilenetv3_large_nonbipfn import create_mobilenetv3_large_ssd
from model.proposed_model import create_mobilenetv3_fcos
from model.efficientnet import get_retinanet_model
from data.loader import loaders
from config.config import train_cfg, data_cfg
from utils.trainer import Trainer

if __name__ == '__main__':
  freeze_support()
  # Create model and get dataloaders
  model = get_retinanet_model(num_classes=data_cfg.num_classes)
  train_loader, val_loader, test_loader = loaders()

  # Initialize trainer
  trainer = Trainer(model, train_loader, val_loader, test_loader, train_cfg)

  # Start training
  final_metrics = trainer.train()