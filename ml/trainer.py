import torch
import torch.nn.functional as F
from torchmetrics import Precision, Recall, F1Score, AUROC, ConfusionMatrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import os
import config

class Trainer:
    # Parameter order: model, train_loader, val_loader, optimizer, criterion, device, scheduler, num_classes=4
    def __init__(self, model, train_loader, val_loader, optimizer, criterion, device, scheduler, num_classes=4):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.scheduler = scheduler  # CosineAnnealingWarmRestarts instance
        self.num_classes = num_classes
        
        # Initialize torchmetrics for multiclass classification, using task="multiclass"
        self.precision_metric = Precision(num_classes=num_classes, average='macro', task="multiclass").to(device)
        self.recall_metric    = Recall(num_classes=num_classes, average='macro', task="multiclass").to(device)
        self.f1_metric        = F1Score(num_classes=num_classes, average='macro', task="multiclass").to(device)
        self.auroc_metric     = AUROC(num_classes=num_classes, average='macro', task="multiclass").to(device)
        self.confusion_matrix_metric = ConfusionMatrix(num_classes=num_classes, task="multiclass").to(device)
        
        self.current_epoch = 0  # To keep track of the epoch for scheduler updates
    
    def train_epoch(self):
        self.model.train()
        running_loss = 0.0
        num_batches = len(self.train_loader)
        # Wrap train_loader with tqdm for progress display
        for batch_idx, (images, labels) in enumerate(tqdm(self.train_loader, desc=f"Epoch {self.current_epoch+1} Training", leave=False)):
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            
            # Update scheduler on a per-batch basis for CosineAnnealingWarmRestarts:
            self.scheduler.step(self.current_epoch + batch_idx / num_batches)
            
            running_loss += loss.item() * images.size(0)
        epoch_loss = running_loss / len(self.train_loader.dataset)
        return epoch_loss
    
    def validate_epoch(self):
        self.model.eval()
        running_loss = 0.0

        # Reset torchmetrics states
        self.precision_metric.reset()
        self.recall_metric.reset()
        self.f1_metric.reset()
        self.auroc_metric.reset()
        self.confusion_matrix_metric.reset()

        # Loop over the validation set without accumulating all predictions
        for images, labels in tqdm(self.val_loader, desc="Validation", leave=False):
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            with torch.no_grad():
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
            
            running_loss += loss.item() * images.size(0)
            
            # Get probabilities and predictions
            probs = F.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)
            
            # Update metrics per batch (incrementally)
            self.precision_metric.update(preds, labels)
            self.recall_metric.update(preds, labels)
            self.f1_metric.update(preds, labels)
            self.auroc_metric.update(probs, labels)
            self.confusion_matrix_metric.update(preds, labels)
        
        # Compute epoch-level metrics (the stateful metrics handle the accumulation)
        epoch_loss = running_loss / len(self.val_loader.dataset)
        precision = self.precision_metric.compute().item()
        recall    = self.recall_metric.compute().item()
        f1        = self.f1_metric.compute().item()
        auroc     = self.auroc_metric.compute().item()
        cm        = self.confusion_matrix_metric.compute().cpu().numpy()
        
        # (Optional) Clear any cached memory
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
        
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
        for epoch in range(num_epochs):
            self.current_epoch = epoch  # update epoch count
            print(f"Epoch {epoch+1}/{num_epochs}")
            train_loss = self.train_epoch()
            val_loss, precision, recall, f1, auroc, cm = self.validate_epoch()
            print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                  f"Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f} | AUROC: {auroc:.4f}")
            self.plot_confusion_matrix(cm, epoch+1)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_model(epoch+1, best_val_loss)
            # No need for an extra scheduler.step() here since it is updated per batch.
