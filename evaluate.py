import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import os
from datetime import datetime
from torch.cuda.amp import autocast
from sklearn.metrics import precision_recall_curve, average_precision_score
from config import DEVICE, MODEL_DIR, SEQUENCE_LENGTH, INPUT_CHANNELS

class TyphoonEvaluator:
    def __init__(self, model, test_loader, criterion, save_dir=None):
        self.model = model.to(DEVICE)
        self.test_loader = test_loader
        self.criterion = criterion
        
        if save_dir is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            self.save_dir = os.path.join(MODEL_DIR, 'evaluations', f'eval_{timestamp}')
        else:
            self.save_dir = save_dir
            
        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(os.path.join(self.save_dir, 'predictions'), exist_ok=True)

    def evaluate(self):
        """Évaluation complète du modèle"""
        self.model.eval()
        total_metrics = {
            'loss': 0,
            'iou': 0,
            'pixel_acc': 0,
            'precision': 0,
            'recall': 0,
            'f1': 0
        }
        
        all_preds = []
        all_targets = []
        n_batches = len(self.test_loader)
        
        with torch.no_grad():
            with tqdm(self.test_loader, desc='Evaluating') as pbar:
                for batch_idx, (inputs, targets) in enumerate(pbar):
                    inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
                    
                    with autocast():
                        outputs = self.model(inputs)
                        loss = self.criterion(outputs, targets)
                    
                    metrics = self.compute_metrics(outputs, targets)
                    
                    for key in total_metrics:
                        total_metrics[key] += metrics[key]
                    
                    all_preds.extend(outputs.cpu().numpy())
                    all_targets.extend(targets.cpu().numpy())
                    
                    if batch_idx % 20 == 0:
                        self.save_prediction_visualization(
                            inputs[0], outputs[0], targets[0], batch_idx
                        )
                    
                    pbar.set_postfix({k: f'{v/(batch_idx+1):.4f}' for k, v in total_metrics.items()})

        avg_metrics = {k: v/n_batches for k, v in total_metrics.items()}
        
        self.analyze_results(np.array(all_preds), np.array(all_targets))
        
        return avg_metrics

    def compute_metrics(self, pred, target, threshold=0.5):
        """Calcul des métriques pour une batch"""
        pred_binary = (pred > threshold).float()
        
        # IoU
        intersection = (pred_binary * target).sum((1, 2, 3))
        union = pred_binary.sum((1, 2, 3)) + target.sum((1, 2, 3)) - intersection
        iou = (intersection / (union + 1e-6)).mean()
        
        # Pixel Accuracy
        correct = (pred_binary == target).float().sum((1, 2, 3))
        total = torch.ones_like(target).sum((1, 2, 3))
        accuracy = (correct / total).mean()
        
        # Precision, Recall, F1
        tp = (pred_binary * target).sum((1, 2, 3))
        fp = pred_binary.sum((1, 2, 3)) - tp
        fn = target.sum((1, 2, 3)) - tp
        
        precision = (tp / (tp + fp + 1e-6)).mean()
        recall = (tp / (tp + fn + 1e-6)).mean()
        f1 = 2 * (precision * recall) / (precision + recall + 1e-6)
        
        return {
            'loss': self.criterion(pred, target).item(),
            'iou': iou.item(),
            'pixel_acc': accuracy.item(),
            'precision': precision.item(),
            'recall': recall.item(),
            'f1': f1.item()
        }

    def save_prediction_visualization(self, input_data, pred, target, batch_idx):
        fig, axes = plt.subplots(3, 3, figsize=(15, 15))
        fig.suptitle('Typhoon Prediction Analysis', fontsize=16)
        
        im0 = axes[0, 0].imshow(input_data[0, 0].cpu())
        axes[0, 0].set_title('Input (U wind)')
        plt.colorbar(im0, ax=axes[0, 0])
        
        for i in range(SEQUENCE_LENGTH):
            row, col = (i+1)//3, (i+1)%3
            pred_frame = pred[i].cpu()
            target_frame = target[i].cpu()
            
            overlay = np.zeros((*pred_frame.shape, 3))
            overlay[..., 0] = pred_frame  # Rouge pour les prédictions
            overlay[..., 1] = target_frame  # Vert pour les cibles
            
            axes[row, col].imshow(overlay)
            axes[row, col].set_title(f'Frame {i+1}\nRed: Pred, Green: Target')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'predictions', f'pred_{batch_idx}.png'))
        plt.close()

    def analyze_results(self, all_preds, all_targets):
        # Courbe Precision-Recall
        plt.figure(figsize=(10, 10))
        precision, recall, _ = precision_recall_curve(
            all_targets.ravel(), 
            all_preds.ravel()
        )
        ap = average_precision_score(all_targets.ravel(), all_preds.ravel())
        
        plt.plot(recall, precision, label=f'AP={ap:.3f}')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend()
        plt.savefig(os.path.join(self.save_dir, 'precision_recall_curve.png'))
        plt.close()
        
        # Distribution des prédictions
        plt.figure(figsize=(10, 5))
        sns.histplot(data=all_preds.ravel(), bins=50)
        plt.title('Distribution of Predictions')
        plt.savefig(os.path.join(self.save_dir, 'pred_distribution.png'))
        plt.close()

def evaluate_model(model_path, test_loader):
    model = torch.load(model_path)
    model.eval()
    
    criterion = nn.BCEWithLogitsLoss()
    
    evaluator = TyphoonEvaluator(model, test_loader, criterion)
    
    metrics = evaluator.evaluate()
    
    print("\nEvaluation Results:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    return metrics, evaluator.save_dir

