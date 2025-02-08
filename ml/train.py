import torch
from torchvision import transforms
from loader import get_loaders
from model import LightCNN
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
    
    model = LightCNN(num_classes=config.NUM_CLASSES).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE)
    
    # For multiclass classification
    criterion = torch.nn.CrossEntropyLoss()
    
    trainer = Trainer(model, train_loader, val_loader, optimizer, criterion, device, num_classes=config.NUM_CLASSES)
    trainer.train(config.NUM_EPOCHS)
    
if __name__ == "__main__":
    main()
