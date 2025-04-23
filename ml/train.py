import torch
from torchvision import transforms
from loader import get_loaders
from model import EfficientNetClassifier
from trainer import Trainer
import config

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Define transformations for training and validation
    train_transform = transforms.Compose([
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.MEAN, std=config.STD)
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.MEAN, std=config.STD)
    ])
    
    train_loader, val_loader = get_loaders(
        data_dir=config.DATA_DIR,
        batch_size=config.BATCH_SIZE,
        train_ratio=config.TRAIN_RATIO,
        transform_train=train_transform,
        transform_val=val_transform
    )
    
    # Initialize the model with transfer learning (pretrained weights) for small datasets
    model = EfficientNetClassifier(num_classes=config.NUM_CLASSES,).to(device)
    
    # Use a lower learning rate for fine-tuning
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    
    # For multiclass classification with class imbalance handling via weighted loss
    class_weights = torch.tensor(config.CLASS_WEIGHTS).float().to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    
    # Scheduler with a shorter restart period for more frequent learning rate adjustments
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
      optimizer,
      T_0=5,
      T_mult=1,
      eta_min=1e-6
    )
    
    trainer = Trainer(model, train_loader, val_loader, optimizer, criterion, device, scheduler, num_classes=config.NUM_CLASSES)
    trainer.train(config.NUM_EPOCHS)
    
if __name__ == "__main__":
    main()