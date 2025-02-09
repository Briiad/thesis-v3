import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import os
from tqdm import tqdm
import wandb  # assuming you're using wandb for logging
from torchmetrics.detection import MeanAveragePrecision
from torchmetrics.classification import MulticlassPrecision, MulticlassRecall, MulticlassF1Score, MulticlassConfusionMatrix

# Import your custom metric (if any) or use FCOS loss outputs as provided by torchvision.
# Here we assume your FCOS model returns a loss dictionary during training.
# Also, assume that your dataloaders provide (images, targets) in the proper format.

class Trainer:
    def __init__(self, model, train_loader, val_loader, config):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config  # config contains epochs, learning_rate, checkpoint_dir, etc.
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        
        # Enable training for backbone if needed
        for param in self.model.backbone.parameters():
            param.requires_grad = True
        
        # Setup optimizer and scheduler
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=config.epochs,
            eta_min=1e-6
        )
        
        # Setup metrics
        self.map_metric = MeanAveragePrecision(
            iou_type='bbox',
            box_format='xyxy',
            average='macro',
            class_metrics=True,
            iou_thresholds=[0.2, 0.25, 0.3, 0.35, 0.4],
            rec_thresholds=[0.2, 0.25, 0.3, 0.35, 0.4]
        ).to(self.device)
        
        # Example: assume 7 classes (adjust if needed)
        self.precision_metric = MulticlassPrecision(num_classes=config.num_classes, average='macro').to(self.device)
        self.recall_metric = MulticlassRecall(num_classes=config.num_classes, average='macro').to(self.device)
        self.f1_metric = MulticlassF1Score(num_classes=config.num_classes, average='macro').to(self.device)
        self.confusion_matrix_metric = MulticlassConfusionMatrix(num_classes=config.num_classes).to(self.device)
        
        os.makedirs(config.checkpoint_dir, exist_ok=True)
        self.best_map = 0.0

    def train_one_epoch(self, epoch: int):
        self.model.train()
        total_loss = 0
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}")
        for images, targets in pbar:
            images = [img.to(self.device) for img in images]
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
            loss_dict = self.model(images, targets)
            loss = sum(loss for loss in loss_dict.values())
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            pbar.set_postfix(loss=loss.item())
        return total_loss / len(self.train_loader)

    def validate(self):
        self.model.eval()
        self.map_metric.reset()
        self.precision_metric.reset()
        self.recall_metric.reset()
        self.f1_metric.reset()
        self.confusion_matrix_metric.reset()
        
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for images, targets in tqdm(self.val_loader, desc="Validating"):
                images = [img.to(self.device) for img in images]
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
                preds = self.model(images)
                all_preds.extend(preds)
                all_targets.extend(targets)
                # For classification metrics, you might compute a “most confident” label per image.
                # (Assume a helper function _prepare_classification_inputs is defined similarly to your original code.)
        
        map_results = self.map_metric(all_preds, all_targets)
        precision = self.precision_metric.compute()
        recall = self.recall_metric.compute()
        f1_score = self.f1_metric.compute()
        confusion_matrix = self.confusion_matrix_metric.compute().cpu().tolist()
        
        # Reset metrics after computation
        self.map_metric.reset()
        self.precision_metric.reset()
        self.recall_metric.reset()
        self.f1_metric.reset()
        self.confusion_matrix_metric.reset()
        
        print(f"Validation mAP: {map_results['map']}")
        print(f"Validation Precision (macro): {precision.mean().item()}")
        print(f"Validation Recall (macro): {recall.mean().item()}")
        print(f"Validation F1 (macro): {f1_score.mean().item()}")
        
        return {
            'val_mAP': map_results['map'],
            'val_precision_macro': precision.mean().item(),
            'val_recall_macro': recall.mean().item(),
            'val_f1_macro': f1_score.mean().item(),
            'val_confusion_matrix': confusion_matrix
        }

    def save_checkpoint(self, epoch: int, metrics: dict, is_best: bool = False):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics
        }
        path = os.path.join(self.config.checkpoint_dir, f'checkpoint_epoch_{epoch}.pth')
        torch.save(checkpoint, path)
        if is_best:
            best_path = os.path.join(self.config.checkpoint_dir, 'best_model.pth')
            torch.save(checkpoint, best_path)

    def train(self):
        wandb.init(project="object_detection", config=self.config)
        for epoch in range(self.config.epochs):
            train_loss = self.train_one_epoch(epoch)
            val_metrics = self.validate()
            self.scheduler.step()
            metrics = {**{'train_loss': train_loss}, **val_metrics}
            wandb.log(metrics)
            if val_metrics['val_mAP'] > self.best_map:
                self.best_map = val_metrics['val_mAP']
                self.save_checkpoint(epoch, metrics, is_best=True)
            self.save_checkpoint(epoch, metrics)
        wandb.finish()