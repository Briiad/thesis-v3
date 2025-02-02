import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
import os
from tqdm import tqdm
from torch.nn.utils import clip_grad_norm_
from torchmetrics.detection import MeanAveragePrecision
from torchmetrics.classification import MulticlassPrecision, MulticlassRecall, MulticlassF1Score, MulticlassConfusionMatrix
import wandb
from typing import Dict
from utils.customMetrics import calculate_map

class Trainer:
    def __init__(self, model, train_loader, val_loader, test_loader, config):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.config = config
        
        # Setup device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        
        for param in self.model.backbone.parameters():
            param.requires_grad = True
        
        # Setup optimizer and scheduler
        self.optimizer = Adam(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        self.scheduler = StepLR(
            optimizer=self.optimizer,
            step_size=config.lr_scheduler_step,
            gamma=config.lr_scheduler_gamma
        )
        
        # Setup metrics
        self.map_metric = MeanAveragePrecision(
            iou_type='bbox',
            box_format='xywh',
            average='macro',
            iou_thresholds=[0.2, 0.25, 0.3, 0.35, 0.4],
            rec_thresholds=[0.2, 0.25, 0.3, 0.35, 0.4]
        ).to(self.device)
        
        # Assuming 7 classes - adjust num_classes as needed
        self.precision_metric = MulticlassPrecision(num_classes=config.num_classes, average='macro').to(self.device)
        self.recall_metric = MulticlassRecall(num_classes=config.num_classes, average='macro').to(self.device)
        self.f1_metric = MulticlassF1Score(num_classes=config.num_classes, average='macro').to(self.device)
        self.confusion_matrix_metric = MulticlassConfusionMatrix(
            num_classes=config.num_classes
        ).to(self.device)
        
        # Create checkpoint directory if it doesn't exist
        os.makedirs(config.checkpoint_dir, exist_ok=True)
        
        # Initialize best mAP for model saving
        self.best_map = 0.0

    def train_one_epoch(self, epoch: int) -> Dict:
        self.model.train()
        total_loss = 0
        progress_bar = tqdm(self.train_loader, desc=f'Epoch {epoch}')
        
        for images, targets in progress_bar:
            # Move images to device
            images = [image.to(self.device) for image in images]
            
            # Move targets to device
            targets = [{k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                      for k, v in t.items()} for t in targets]
            
            # Forward pass
            loss_dict = self.model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            
            # Backward pass
            self.optimizer.zero_grad()
            losses.backward()
            clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += losses.item()
            progress_bar.set_postfix({'loss': losses.item()})
        
        avg_loss = total_loss / len(self.train_loader)
        return {'train_loss': avg_loss}
      
    def _prepare_classification_inputs(self, predictions, targets):
        """
        Prepare inputs for classification metrics from object detection predictions.
        
        Args:
            predictions (List[Dict]): Model predictions
            targets (List[Dict]): Ground truth targets
        
        Returns:
            Tuple of tensors for classification metrics
        """
        # Collect all predicted labels and true labels
        all_pred_labels = []
        all_true_labels = []
        
        for pred, target in zip(predictions, targets):
            # If using the most confident prediction per image
            if len(pred['labels']) > 0:
                # Find the index of the highest scoring prediction
                max_score_idx = pred['scores'].argmax()
                all_pred_labels.append(pred['labels'][max_score_idx])
            else:
                # If no predictions, use a background/null class (typically 0)
                all_pred_labels.append(torch.tensor(0).to(self.device))
            
            # Use the first/most confident label from ground truth if multiple
            if len(target['labels']) > 0:
                all_true_labels.append(target['labels'][0])
            else:
                # If no ground truth labels, use a background/null class
                all_true_labels.append(torch.tensor(0).to(self.device))
        
        # Convert to tensor
        pred_labels = torch.stack(all_pred_labels)
        true_labels = torch.stack(all_true_labels)
        
        return pred_labels, true_labels

    def validate(self) -> Dict:
        """Validate the model using custom mAP implementation"""
        self.model.eval()
        
        # Reset classification metrics
        self.precision_metric.reset()
        self.recall_metric.reset()
        self.f1_metric.reset()
        self.confusion_matrix_metric.reset()
        
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for images, targets in tqdm(self.val_loader, desc='Validating'):
                images = [image.to(self.device) for image in images]
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
                predictions = self.model(images)
                
                all_preds.extend(predictions)
                all_targets.extend(targets)
        
        # Calculate mAP using custom implementation
        map_results = calculate_map(all_preds, all_targets, num_classes=self.config.num_classes)
        map_torch_results = self.map_metric(all_preds, all_targets)
        
        # Calculate classification metrics
        pred_labels, true_labels = self._prepare_classification_inputs(all_preds, all_targets)
        self.precision_metric.update(pred_labels, true_labels)
        self.recall_metric.update(pred_labels, true_labels)
        self.f1_metric.update(pred_labels, true_labels)
        self.confusion_matrix_metric.update(pred_labels, true_labels)
        
        # Compute classification metrics
        precision = self.precision_metric.compute()
        recall = self.recall_metric.compute()
        f1_score = self.f1_metric.compute()
        confusion_matrix = self.confusion_matrix_metric.compute().cpu().tolist()
        
        # Print validation results
        print(f"Validation mAP: {map_results['map']}")
        print(f"Validation mAP (Torch): {map_torch_results}")
        print(f"Validation Precision: {precision}")
        print(f"Validation Recall: {recall}")
        print(f"Validation F1 Score: {f1_score}")
        
        wandb.log({"val_confusion_matrix": wandb.plot.confusion_matrix(
            preds=pred_labels.cpu().numpy(),
            y_true=true_labels.cpu().numpy(),
            class_names=[f"Class_{i}" for i in range(self.config.num_classes)]
        )})
        
        # Reset classification metrics
        self.map_metric.reset()
        self.precision_metric.reset()
        self.recall_metric.reset()
        self.f1_metric.reset()
        self.confusion_matrix_metric.reset()
        
        return {
            'val_mAP': map_results['map'],
            'val_mAP_torch': map_torch_results['map'].item(),
            'val_confusion_matrix': confusion_matrix,
            'val_precision_per_class': precision.tolist(),
            'val_recall_per_class': recall.tolist(),
            'val_f1_score_per_class': f1_score.tolist(),
            'val_precision_macro': precision.mean().item(),
            'val_recall_macro': recall.mean().item(),
            'val_f1_score_macro': f1_score.mean().item()
        }

    def test(self) -> Dict:
        """Test the model using custom mAP implementation"""
        self.model.eval()
        
        # Reset classification metrics
        self.precision_metric.reset()
        self.recall_metric.reset()
        self.f1_metric.reset()
        self.confusion_matrix_metric.reset()
        
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for images, targets in tqdm(self.test_loader, desc='Testing'):
                images = [image.to(self.device) for image in images]
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
                predictions = self.model(images)
                
                all_preds.extend(predictions)
                all_targets.extend(targets)
        
        # Calculate mAP using custom implementation
        map_results = calculate_map(all_preds, all_targets, num_classes=self.config.num_classes)
        map_torch_results = self.map_metric(all_preds, all_targets)
        
        # Calculate classification metrics
        pred_labels, true_labels = self._prepare_classification_inputs(all_preds, all_targets)
        self.precision_metric.update(pred_labels, true_labels)
        self.recall_metric.update(pred_labels, true_labels)
        self.f1_metric.update(pred_labels, true_labels)
        self.confusion_matrix_metric.update(pred_labels, true_labels)
        
        # Compute classification metrics
        precision = self.precision_metric.compute()
        recall = self.recall_metric.compute()
        f1_score = self.f1_metric.compute()
        confusion_matrix = self.confusion_matrix_metric.compute().cpu().tolist()
        
        # Reset classification metrics
        self.map_metric.reset()
        self.precision_metric.reset()
        self.recall_metric.reset()
        self.f1_metric.reset()
        self.confusion_matrix_metric.reset()
        
        return {
            'test_mAP': map_results['map'],
            'test_mAP_torch': map_torch_results['map'].item(),
            'test_confusion_matrix': confusion_matrix,
            'test_precision_per_class': precision.tolist(),
            'test_recall_per_class': recall.tolist(),
            'test_f1_score_per_class': f1_score.tolist(),
            'test_precision_macro': precision.mean().item(),
            'test_recall_macro': recall.mean().item(),
            'test_f1_score_macro': f1_score.mean().item()
        }
    
    def save_checkpoint(self, epoch: int, metrics: Dict, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics
        }
        
        # Save regular checkpoint
        if epoch % self.config.save_frequency == 0:
            path = os.path.join(self.config.checkpoint_dir, f'checkpoint_epoch_{epoch}.pth')
            torch.save(checkpoint, path)
        
        # Save best model
        if is_best:
            best_path = os.path.join(self.config.checkpoint_dir, 'best_model.pth')
            torch.save(checkpoint, best_path)

    def train(self):
        """Full training loop"""
        # Initialize wandb
        wandb.init(project="object_detection", config=self.config)
        
        for epoch in range(self.config.epochs):
            # Training
            train_metrics = self.train_one_epoch(epoch)
            
            # Validation
            val_metrics = self.validate()
            
            # Update learning rate
            self.scheduler.step()
            
            # Log metrics
            metrics = {**train_metrics, **val_metrics}
            wandb.log(metrics)
            
            # Save checkpoint if it's the best model
            if val_metrics['val_mAP'] > self.best_map:
                self.best_map = val_metrics['val_mAP']
                self.save_checkpoint(epoch, metrics, is_best=True)
            
            # Regular checkpoint saving
            self.save_checkpoint(epoch, metrics)
        
        # Final test
        test_metrics = self.test()
        wandb.log(test_metrics)
        wandb.finish()
        
        return test_metrics