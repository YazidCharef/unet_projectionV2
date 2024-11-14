import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import logging
from tqdm import tqdm
from datetime import datetime
from torch.cuda.amp import autocast, GradScaler
from config import DEVICE, NUM_GPUS, MODEL_DIR

class MetricTracker:
    def __init__(self):
        self.reset()

    def reset(self):
        self.loss = 0
        self.iou = 0
        self.pixel_acc = 0
        self.count = 0

    def update(self, loss, pred_masks, true_masks):
        batch_size = pred_masks.size(0)
        self.count += batch_size
        self.loss += loss.item() * batch_size
        
        # Calcul IoU
        pred_binary = (pred_masks > 0.5).float()
        intersection = (pred_binary * true_masks).sum((1, 2, 3))
        union = pred_binary.sum((1, 2, 3)) + true_masks.sum((1, 2, 3)) - intersection
        iou = (intersection / (union + 1e-6)).mean()
        self.iou += iou.item() * batch_size

        # Calcul Pixel Accuracy
        correct = (pred_binary == true_masks).float().sum((1, 2, 3))
        total = torch.ones_like(true_masks).sum((1, 2, 3))
        accuracy = (correct / total).mean()
        self.pixel_acc += accuracy.item() * batch_size

    def get_metrics(self):
        return {
            'loss': self.loss / self.count,
            'iou': self.iou / self.count,
            'pixel_acc': self.pixel_acc / self.count
        }

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = os.path.join(MODEL_DIR, f'run_{timestamp}')
    os.makedirs(log_dir, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, 'training.log')),
            logging.StreamHandler()
        ]
    )

    model = model.to(DEVICE)
    
    scaler = GradScaler()
    
    train_tracker = MetricTracker()
    val_tracker = MetricTracker()
    
    # Best model tracking
    best_val_loss = float('inf')
    best_model_path = os.path.join(log_dir, 'best_model.pth')
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_tracker.reset()
        
        with tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]') as pbar:
            for batch_idx, (inputs, targets) in enumerate(pbar):
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
                
                optimizer.zero_grad()
                
                with autocast():
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                
                # Backward pass with gradient scaling
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                
                # Update metrics
                train_tracker.update(loss, outputs.detach(), targets)
                
                # Update progress bar
                metrics = train_tracker.get_metrics()
                pbar.set_postfix({
                    'loss': f"{metrics['loss']:.4f}",
                    'iou': f"{metrics['iou']:.4f}",
                    'acc': f"{metrics['pixel_acc']:.4f}"
                })
        
        # Validation phase
        model.eval()
        val_tracker.reset()
        
        with torch.no_grad():
            with tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]') as pbar:
                for inputs, targets in pbar:
                    inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
                    
                    with autocast():
                        outputs = model(inputs)
                        loss = criterion(outputs, targets)
                    
                    val_tracker.update(loss, outputs, targets)
                    
                    metrics = val_tracker.get_metrics()
                    pbar.set_postfix({
                        'loss': f"{metrics['loss']:.4f}",
                        'iou': f"{metrics['iou']:.4f}",
                        'acc': f"{metrics['pixel_acc']:.4f}"
                    })
        
        # Log metrics
        train_metrics = train_tracker.get_metrics()
        val_metrics = val_tracker.get_metrics()
        
        log_msg = (f"\nEpoch {epoch+1}/{num_epochs}\n"
                  f"Train - Loss: {train_metrics['loss']:.4f}, IoU: {train_metrics['iou']:.4f}, "
                  f"Acc: {train_metrics['pixel_acc']:.4f}\n"
                  f"Val - Loss: {val_metrics['loss']:.4f}, IoU: {val_metrics['iou']:.4f}, "
                  f"Acc: {val_metrics['pixel_acc']:.4f}")
        logging.info(log_msg)
        
        # Scheduler step
        scheduler.step(val_metrics['loss'])
        
        # Save best model
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict() if NUM_GPUS <= 1 else model.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': best_val_loss,
                'metrics': val_metrics,
            }, best_model_path)
            logging.info(f"New best model saved with validation loss: {best_val_loss:.4f}")
        
        # Save checkpoint
        if (epoch + 1) % 5 == 0:
            checkpoint_path = os.path.join(log_dir, f'checkpoint_epoch_{epoch+1}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict() if NUM_GPUS <= 1 else model.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'metrics': val_metrics,
            }, checkpoint_path)
            logging.info(f"Checkpoint saved at epoch {epoch+1}")

    return model, best_model_path