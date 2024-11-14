import torch
import torch.nn as nn
import torch.optim as optim
import logging
import os
from datetime import datetime

from config import (
    DEVICE, BATCH_SIZE, NUM_EPOCHS, LEARNING_RATE,
    INPUT_CHANNELS, SEQUENCE_LENGTH, MODEL_DIR
)
from data_processing import get_data_loaders
from models.unet import TemporalUNet
from train import train_model
from evaluate import evaluate_model

def setup_logging(run_dir):
    os.makedirs(run_dir, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(run_dir, 'training.log')),
            logging.StreamHandler()
        ]
    )

def main():
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = os.path.join(MODEL_DIR, f'run_{timestamp}')
    setup_logging(run_dir)

    try:
        print("Loading data...")
        train_loader, val_loader, test_loader = get_data_loaders(
            batch_size=BATCH_SIZE,
            data_fraction=0.1  # for fast testing, 1.0 for full testing
        )
        print(f"Data loaded - Training batches: {len(train_loader)}")

        print("\nInitializing model...")
        model = TemporalUNet(
            n_channels=INPUT_CHANNELS,
            n_frames=SEQUENCE_LENGTH
        ).to(DEVICE)

        n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Model parameters: {n_parameters:,}")

        # 3. Configuration entrainement
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )

        print("\nStarting training...")
        trained_model, best_model_path = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            num_epochs=NUM_EPOCHS
        )

        # 5. evaluation
        print("\nEvaluating model...")
        metrics, eval_dir = evaluate_model(
            model_path=best_model_path,
            test_loader=test_loader
        )

        print("\nFinal Results:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")

        results = {
            'model_path': best_model_path,
            'eval_dir': eval_dir,
            'metrics': metrics
        }
        
        with open(os.path.join(run_dir, 'results.txt'), 'w') as f:
            f.write(f"Training completed at: {datetime.now()}\n\n")
            f.write(f"Model parameters: {n_parameters:,}\n")
            f.write("\nMetrics:\n")
            for metric, value in metrics.items():
                f.write(f"{metric}: {value:.4f}\n")
            f.write(f"\nBest model saved at: {best_model_path}")

        return results

    except Exception as e:
        print(f"\nError occurred: {str(e)}")
        raise

if __name__ == "__main__":
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    print(f"Using device: {DEVICE}")
    print(f"Starting training at: {datetime.now()}\n")

    try:
        results = main()
        print("\nTraining completed successfully!")
        print(f"Results saved in: {results['eval_dir']}")
        
    except Exception as e:
        print(f"Training failed: {str(e)}")
        raise