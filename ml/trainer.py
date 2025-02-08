import torch
import torch.nn.functional as F
from torchmetrics import Precision, Recall, F1Score, AUROC, ConfusionMatrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from tqdm import tqdm  # Import tqdm for progress bars
import config

class Trainer:
    def __init__(self, model, train_loader, val_loader, optimizer, criterion, device, num_classes=4):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.num_classes = num_classes
        
        # Initialize torchmetrics for multiclass classification
        self.precision_metric = Precision(num_classes=num_classes, average='macro', task='multiclass').to(device)
        self.recall_metric    = Recall(num_classes=num_classes, average='macro', task='multiclass').to(device)
        self.f1_metric        = F1Score(num_classes=num_classes, average='macro', task='multiclass').to(device)
        self.auroc_metric     = AUROC(num_classes=num_classes, average='macro', task='multiclass').to(device)
        self.confusion_matrix_metric = ConfusionMatrix(num_classes=num_classes, task='multiclass').to(device)
    
    def train_epoch(self):
        self.model.train()
        running_loss = 0.0
        # Wrap the training DataLoader with tqdm progress bar
        for images, labels in tqdm(self.train_loader, desc="Training", leave=False):
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item() * images.size(0)
        epoch_loss = running_loss / len(self.train_loader.dataset)
        return epoch_loss
    
    def validate_epoch(self):
        self.model.eval()
        running_loss = 0.0
        all_preds = []
        all_labels = []
        all_probs = []
        
        # Wrap the validation DataLoader with tqdm progress bar
        with torch.no_grad():
            for images, labels in tqdm(self.val_loader, desc="Validation", leave=False):
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                running_loss += loss.item() * images.size(0)
                
                # Get predictions and probabilities
                probs = F.softmax(outputs, dim=1)
                preds = torch.argmax(probs, dim=1)
                
                all_preds.append(preds.cpu())
                all_labels.append(labels.cpu())
                all_probs.append(probs.cpu())
        
        epoch_loss = running_loss / len(self.val_loader.dataset)
        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)
        all_probs = torch.cat(all_probs)
        
        # Compute metrics
        precision = self.precision_metric(all_preds, all_labels).item()
        recall    = self.recall_metric(all_preds, all_labels).item()
        f1        = self.f1_metric(all_preds, all_labels).item()
        auroc     = self.auroc_metric(all_probs, all_labels).item()
        
        # Compute confusion matrix
        cm = self.confusion_matrix_metric(all_preds, all_labels).cpu().numpy()
        
        return epoch_loss, precision, recall, f1, auroc, cm
    
    def plot_confusion_matrix(self, cm, epoch):
        plt.figure(figsize=(8,6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title(f'Confusion Matrix at Epoch {epoch}')
        plt.savefig(f'confusion_matrix_epoch_{epoch}.png')
        plt.close()
    
    def save_model(self, epoch, best_metric):
        # Save model checkpoint
        import os
        save_dir = os.path.dirname(config.MODEL_SAVE_PATH)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_metric': best_metric,
        }, config.MODEL_SAVE_PATH)
        print(f"Model saved to {config.MODEL_SAVE_PATH}")
    
    def train(self, num_epochs):
        best_val_loss = float('inf')
        for epoch in range(1, num_epochs + 1):
            print(f"Epoch {epoch}/{num_epochs}")
            train_loss = self.train_epoch()
            val_loss, precision, recall, f1, auroc, cm = self.validate_epoch()
            print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                  f"Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f} | AUROC: {auroc:.4f}")
            self.plot_confusion_matrix(cm, epoch)
            
            # Optionally save model if validation loss decreases
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_model(epoch, best_val_loss)
