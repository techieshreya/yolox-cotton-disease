import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import time
import logging
import datetime

# Local imports
from utils.dataset import CottonDiseaseDataset, get_train_augs, get_val_augs
from utils.loss import YOLOXLoss, severity_aware_loss, multi_task_loss
from models.yolox_mpp import YOLOXMPP

def generate_run_id():
    """Generate a unique run ID based on current timestamp"""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"yolox_mpp_run_{timestamp}"

def setup_logging(save_dir):
    """Setup logging to both console and file"""
    logs_dir = os.path.join(save_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    log_file = os.path.join(logs_dir, "training.log")

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def collate_fn(batch):
    """Custom collate function for handling variable number of targets"""
    images, targets = zip(*batch)
    return torch.stack(images), list(targets)

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Generate unique run ID
    run_id = generate_run_id()
    run_save_dir = os.path.join(args.save_dir, run_id)
    os.makedirs(run_save_dir, exist_ok=True)
    
    # Setup logging
    logger = setup_logging(run_save_dir)
    
    logger.info(f"Starting YOLOX-M++ training run: {run_id}")
    logger.info(f"Using device: {device}")
    logger.info(f"Arguments: {args}")
    
    # Create augmentation pipelines
    train_augmentations = get_train_augs(args.input_size)
    val_augmentations = get_val_augs(args.input_size)
    
    # Datasets
    train_dataset = CottonDiseaseDataset(
        data_dir="data/train",
        augmentations=train_augmentations,
        input_size=args.input_size,
    )
    val_dataset = CottonDiseaseDataset(
        data_dir="data/valid",
        augmentations=val_augmentations,
        input_size=args.input_size,
    )
    
    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    
    # Model
    model = YOLOXMPP(num_classes=args.num_classes, phi=args.phi).to(device)
    
    # Optimizer with tuned parameters
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    # Learning rate scheduler
    warmup_epochs = args.warmup_epochs
    lr_warmup_factor = 1.0 / warmup_epochs
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=20,
        T_mult=2,
        eta_min=args.lr * 0.01
    )
    
    # Loss functions
    criterion = YOLOXLoss(
        num_classes=args.num_classes,
        strides=model.stride.tolist(),
        use_focal_loss=True,
        alpha_iou=args.alpha_iou,
    )
    
    # Severity weights (based on class distribution)
    severity_weights = torch.tensor([1.0, 1.0, 1.0, 2.0, 1.0])  # Higher weight for rare classes
    
    # Early stopping
    best_map = 0.0
    patience = args.patience
    patience_counter = 0
    min_delta = args.min_delta
    
    logger.info(f"Training started with {len(train_dataset)} training samples and {len(val_dataset)} validation samples")
    
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        train_detection_loss = 0.0
        train_severity_loss = 0.0
        
        # LR Warmup
        if epoch < warmup_epochs:
            for param_group in optimizer.param_groups:
                param_group["lr"] = args.lr * (epoch + 1) * lr_warmup_factor
        
        # Training loop
        for images, targets_xyxy in tqdm(
            train_loader, desc=f"Train Epoch {epoch + 1}/{args.epochs}"
        ):
            images = images.to(device)
            targets_xyxy = [t.to(device) for t in targets_xyxy]
            
            optimizer.zero_grad()
            outputs = model(images)
            
            # Calculate loss
            total_loss = criterion(outputs, targets_xyxy)
            
            # Add severity-aware loss if available
            if len(outputs) > 0 and outputs[0].shape[1] > args.num_classes + 5:
                # Extract severity predictions (assuming they're concatenated)
                severity_preds = []
                for output in outputs:
                    # Assuming format: [reg, obj, cls, severity]
                    severity_pred = output[:, 5 + args.num_classes:, :, :]
                    severity_preds.append(severity_pred)
                
                # Calculate severity loss (simplified for now)
                severity_loss = 0.0
                for i, target in enumerate(targets_xyxy):
                    if len(target) > 0:
                        # Extract severity labels from class labels
                        severity_labels = target[:, 4].long()  # Assuming severity is encoded in class labels
                        if len(severity_preds) > 0:
                            severity_pred = severity_preds[0][i:i+1]  # Simplified
                            severity_loss += severity_aware_loss(
                                severity_pred.view(-1, 5), 
                                severity_labels, 
                                severity_weights
                            )
                
                total_loss = total_loss + 0.1 * severity_loss
                train_severity_loss += severity_loss.item()
            
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += total_loss.item()
            train_detection_loss += total_loss.item()
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_detection_loss = 0.0
        val_severity_loss = 0.0
        
        with torch.no_grad():
            for images, targets_xyxy in tqdm(val_loader, desc="Validating"):
                images = images.to(device)
                targets_xyxy = [t.to(device) for t in targets_xyxy]
                
                outputs = model(images)
                val_loss_batch = criterion(outputs, targets_xyxy)
                val_loss += val_loss_batch.item()
                val_detection_loss += val_loss_batch.item()
        
        # Scheduler step
        if epoch >= warmup_epochs:
            scheduler.step()
        
        # Calculate average losses
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        avg_train_detection_loss = train_detection_loss / len(train_loader)
        avg_train_severity_loss = train_severity_loss / len(train_loader)
        
        # Logging
        logger.info(
            f"Epoch {epoch + 1}/{args.epochs} | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} | "
            f"Detection Loss: {avg_train_detection_loss:.4f} | "
            f"Severity Loss: {avg_train_severity_loss:.4f} | "
            f"LR: {optimizer.param_groups[0]['lr']:.2e}"
        )
        
        # Early stopping (simplified - using validation loss instead of mAP for now)
        if avg_val_loss < best_map - min_delta:
            best_map = avg_val_loss
            patience_counter = 0
            torch.save(
                model.state_dict(), os.path.join(run_save_dir, "best_model.pth")
            )
            logger.info(f"New best model saved with Val Loss: {best_map:.4f}")
        else:
            patience_counter += 1
            logger.info(f"No improvement for {patience_counter} epochs (patience: {patience})")
            
            if patience_counter >= patience:
                logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                logger.info(f"Best Val Loss achieved: {best_map:.4f}")
                break
        
        # Save checkpoint
        if (epoch + 1) % args.save_interval == 0:
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": avg_val_loss,
                },
                os.path.join(run_save_dir, f"epoch_{epoch + 1}.pth"),
            )
    
    logger.info(f"YOLOX-M++ training completed for run: {run_id}")
    logger.info(f"All checkpoints and logs saved in: {run_save_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLOX-M++ Training Script")
    parser.add_argument(
        "--input_size",
        type=int,
        nargs=2,
        default=[640, 640],
        help="Model input size [height, width]",
    )
    parser.add_argument(
        "--num_classes", type=int, default=5, help="Number of object classes"
    )
    parser.add_argument(
        "--phi", type=str, default="m", help="Model size: 's', 'm', 'l', 'x'"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,  # Reduced for M++ model
        help="Batch size for training",
    )
    parser.add_argument(
        "--epochs", type=int, default=100, help="Number of training epochs"
    )
    parser.add_argument(
        "--lr", type=float, default=5e-4, help="Learning rate (reduced for M++ model)"
    )
    parser.add_argument(
        "--weight_decay", type=float, default=1e-4, help="Weight decay"
    )
    parser.add_argument(
        "--alpha_iou", type=float, default=2.0, help="Alpha parameter for α-IoU loss"
    )
    parser.add_argument(
        "--warmup_epochs", type=int, default=10, help="Learning rate warmup epochs"
    )
    parser.add_argument(
        "--patience", type=int, default=30, help="Early stopping patience"
    )
    parser.add_argument(
        "--min_delta", type=float, default=0.001, help="Minimum improvement threshold"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="Number of workers for data loading",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./checkpoints_mpp",
        help="Directory to save checkpoints",
    )
    parser.add_argument(
        "--save_interval", type=int, default=10, help="Save checkpoint every N epochs"
    )

    try:
        args = parser.parse_args()
    except SystemExit:
        args = parser.parse_args([])

    os.makedirs(args.save_dir, exist_ok=True)

    try:
        train(args)
        print("YOLOX-M++ training completed successfully!")
    except Exception as e:
        print(f"YOLOX-M++ training failed with error: {e}")
        raise
