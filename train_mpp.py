import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import time
import logging
import datetime
import torchvision

# Local imports
from utils.dataset import CottonDiseaseDataset, get_train_augs, get_val_augs
from utils.loss import YOLOXLoss, severity_aware_loss, multi_task_loss
from models.yolox_mpp import YOLOXMPP

def calculate_iou(box1, boxes):
    """Calculate IoU between box1 and boxes"""
    x1 = np.maximum(box1[0], boxes[:, 0])
    y1 = np.maximum(box1[1], boxes[:, 1])
    x2 = np.minimum(box1[2], boxes[:, 2])
    y2 = np.minimum(box1[3], boxes[:, 3])
    intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union = area1 + area2 - intersection
    return intersection / (union + 1e-6)

def calculate_map(pred_results, true_boxes, true_classes, iou_threshold=0.5):
    """Calculate mAP for a single class"""
    if pred_results.shape[0] == 0:
        return 0.0 if true_boxes.shape[0] > 0 else 1.0
    if true_boxes.shape[0] == 0:
        return 0.0

    pred_boxes = pred_results[:, :4]
    pred_scores = pred_results[:, 4]
    pred_classes = pred_results[:, 5]

    sorted_ind = np.argsort(-pred_scores)
    pred_boxes = pred_boxes[sorted_ind]
    pred_classes = pred_classes[sorted_ind]

    tp = np.zeros(len(pred_boxes))
    fp = np.zeros(len(pred_boxes))
    gt_matched = np.zeros(len(true_boxes))

    for i in range(len(pred_boxes)):
        ious = calculate_iou(pred_boxes[i], true_boxes)
        best_gt_idx = np.argmax(ious)
        best_iou = ious[best_gt_idx]

        if best_iou >= iou_threshold and gt_matched[best_gt_idx] == 0:
            tp[i] = 1
            gt_matched[best_gt_idx] = 1
        else:
            fp[i] = 1

    tp_cumsum = np.cumsum(tp)
    fp_cumsum = np.cumsum(fp)
    recalls = tp_cumsum / len(true_boxes)
    precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-6)

    # Calculate AP using 11-point interpolation
    ap = 0
    for t in np.arange(0, 1.1, 0.1):
        if np.sum(recalls >= t) == 0:
            p = 0
        else:
            p = np.max(precisions[recalls >= t])
        ap += p / 11

    return ap

def calculate_map_per_class(pred_results, true_boxes, true_classes, num_classes, iou_threshold=0.5):
    """Calculate mAP for each class"""
    aps = []
    for cls in range(num_classes):
        pred_mask = pred_results[:, 5] == cls
        gt_mask = true_classes == cls
        ap = calculate_map(
            pred_results[pred_mask],
            true_boxes[gt_mask],
            true_classes[gt_mask],
            iou_threshold,
        )
        aps.append(ap)
    return aps

def postprocess(predictions, strides, num_classes, conf_thre=0.25, nms_thre=0.5):
    """
    Post-processes raw predictions from the model.
    'predictions': A list of 3 tensors, one for each FPN level.
                   Each tensor is of shape [N, C, H, W] where N is the total number of images.
    'strides': A tensor of strides for each FPN level.
    """
    all_detections = []
    device = predictions[0].device

    # 1. Decode predictions from all levels
    decoded_preds = []
    for i, preds_level in enumerate(predictions):
        stride = strides[i]
        N, C, H, W = preds_level.shape

        # Create grid
        yv, xv = torch.meshgrid(
            [torch.arange(H, device=device), torch.arange(W, device=device)],
            indexing="ij",
        )
        grid = (
            torch.stack((xv, yv), 2).view(1, H * W, 2).repeat(N, 1, 1)
        )  # Shape: [N, H*W, 2]

        # Reshape and decode
        preds_level = preds_level.permute(0, 2, 3, 1).reshape(
            N, H * W, C
        )  # Shape: [N, H*W, C]

        box_xy = (preds_level[..., :2] + grid) * stride
        box_wh = torch.exp(preds_level[..., 2:4]) * stride

        # Combine decoded boxes with obj and cls scores (still logits)
        decoded_level = torch.cat((box_xy, box_wh, preds_level[..., 4:]), dim=-1)
        decoded_preds.append(decoded_level)

    output = torch.cat(decoded_preds, dim=1)  # Shape: [N_images, total_anchors, 5+C]

    # 2. Loop through images in the batch to perform NMS
    for i in range(output.shape[0]):
        image_preds = output[i]  # Shape: [total_anchors, 5+C]

        # Apply score threshold
        obj_score = image_preds[:, 4].sigmoid()
        cls_scores = image_preds[:, 5:].sigmoid()

        class_conf, class_pred = torch.max(cls_scores, 1)
        conf_mask = (obj_score * class_conf) >= conf_thre

        # Filter predictions
        image_preds = image_preds[conf_mask]
        class_conf = class_conf[conf_mask]
        class_pred = class_pred[conf_mask]

        if len(image_preds) == 0:
            all_detections.append(torch.zeros((0, 6), device=device))
            continue

        # Convert to xyxy format
        box_cxcy = image_preds[:, :2]
        box_wh = image_preds[:, 2:4]
        box_xyxy = torch.zeros_like(box_cxcy)
        box_xyxy[:, 0] = box_cxcy[:, 0] - box_wh[:, 0] / 2  # x1
        box_xyxy[:, 1] = box_cxcy[:, 1] - box_wh[:, 1] / 2  # y1
        box_xyxy[:, 2] = box_cxcy[:, 0] + box_wh[:, 0] / 2  # x2
        box_xyxy[:, 3] = box_cxcy[:, 1] + box_wh[:, 1] / 2  # y2

        # Combine with scores and classes
        detections = torch.cat([
            box_xyxy,
            (obj_score[conf_mask] * class_conf).unsqueeze(1),
            class_pred.unsqueeze(1).float()
        ], dim=1)

        # Apply NMS
        keep_indices = torchvision.ops.nms(
            detections[:, :4], detections[:, 4], nms_thre
        )
        detections = detections[keep_indices]

        all_detections.append(detections)

    return all_detections

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
    
    # DataLoaders with class-balanced sampling
    sampler = None
    if getattr(args, 'balanced_sampling', True):
        # Normalize weights
        weights = np.array(train_dataset.sample_weights, dtype=np.float64)
        weights = weights / weights.sum()
        sampler = WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
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
    
    # Multi-scale training: change input size every N iterations
    multiscale = getattr(args, 'multiscale', True)
    ms_min, ms_max = getattr(args, 'ms_min', 480), getattr(args, 'ms_max', 800)

    # Learning rate scheduler
    warmup_epochs = args.warmup_epochs
    lr_warmup_factor = 1.0 / warmup_epochs
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=20,
        T_mult=2,
        eta_min=args.lr * 0.01
    )

    # EMA setup
    use_ema = getattr(args, 'use_ema', True)
    ema_decay = getattr(args, 'ema_decay', 0.9998)
    ema_state = None
    if use_ema:
        ema_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    def _ema_update():
        if not use_ema:
            return
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                ema_state[k].mul_(ema_decay).add_(v.detach(), alpha=1.0 - ema_decay)
    
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
            # Optionally resize on-the-fly for multi-scale
            if multiscale:
                short_side = np.random.randint(ms_min // 32, ms_max // 32 + 1) * 32
                images = torch.nn.functional.interpolate(
                    images, size=(short_side, short_side), mode='bilinear', align_corners=False
                )

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
            _ema_update()
            
            train_loss += total_loss.item()
            train_detection_loss += total_loss.item()
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_detection_loss = 0.0
        val_severity_loss = 0.0
        per_class_aps = None
        
        with torch.no_grad():
            # Optionally evaluate EMA weights
            orig_state = None
            if use_ema and ema_state is not None:
                orig_state = {k: v.clone() for k, v in model.state_dict().items()}
                model.load_state_dict(ema_state, strict=False)

            all_outputs = []
            all_targets = []
            for images, targets_xyxy in tqdm(val_loader, desc="Validating"):
                images = images.to(device)
                targets_xyxy = [t.to(device) for t in targets_xyxy]
                outputs = model(images)
                val_loss_batch = criterion(outputs, targets_xyxy)
                val_loss += val_loss_batch.item()
                val_detection_loss += val_loss_batch.item()
                all_outputs.append(outputs)
                all_targets.extend(targets_xyxy)

            # Compute mAP@0.5 similar to train.py
            # Flatten outputs across batches per FPN into a single list of levels
            # outputs is a list of tensors [levels], each: [N, C, H, W]
            # Concatenate along N for each level
            per_level = None
            for batch_out in all_outputs:
                if per_level is None:
                    per_level = [bo for bo in batch_out]
                else:
                    per_level = [torch.cat((a, b), dim=0) for a, b in zip(per_level, batch_out)]

            # Postprocess using strides from model
            final_detections = postprocess(
                per_level, model.stride.to(device), args.num_classes, conf_thre=0.25
            )

            per_class_aps = np.zeros(args.num_classes)
            for i in range(len(final_detections)):
                pred_results_np = final_detections[i].detach().cpu().numpy()
                if i < len(all_targets):
                    gt = all_targets[i].detach().cpu().numpy()
                    if gt.shape[0] > 0:
                        gt_boxes = gt[:, :4]
                        gt_classes = gt[:, 4]
                        aps = calculate_map_per_class(
                            pred_results_np, gt_boxes, gt_classes, args.num_classes
                        )
                        per_class_aps += np.array(aps)
            num_gt_images = sum(1 for gt in all_targets if gt.shape[0] > 0)
            if num_gt_images > 0:
                per_class_aps /= num_gt_images
        
        # Restore original weights after EMA eval
        if use_ema and ema_state is not None and orig_state is not None:
            model.load_state_dict(orig_state, strict=False)

        # Scheduler step
        if epoch >= warmup_epochs:
            scheduler.step()
        
        # Calculate average losses
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        avg_train_detection_loss = train_detection_loss / len(train_loader)
        avg_train_severity_loss = train_severity_loss / len(train_loader)
        
        # Logging
        if per_class_aps is not None:
            mean_ap = float(np.mean(per_class_aps))
            curl1_indicator = ("🔴" if per_class_aps[0] < 0.5 else "🟡" if per_class_aps[0] < 0.8 else "🟢")
            logger.info(
                f"Epoch {epoch + 1}/{args.epochs} | "
                f"Train Loss: {avg_train_loss:.4f} | "
                f"Val Loss: {avg_val_loss:.4f} | "
                f"mAP@0.5: {mean_ap:.4f} | "
                f"{curl1_indicator} AP (Curl stage-1): {per_class_aps[0]:.4f} | "
                f"AP (Curl stage-2): {per_class_aps[1]:.4f} | "
                f"AP (Healthy): {per_class_aps[2]:.4f} | "
                f"AP (Leaf Enation): {per_class_aps[3]:.4f} | "
                f"AP (Sooty): {per_class_aps[4]:.4f} | "
                f"LR: {optimizer.param_groups[0]['lr']:.2e}"
            )
        else:
            logger.info(
                f"Epoch {epoch + 1}/{args.epochs} | "
                f"Train Loss: {avg_train_loss:.4f} | "
                f"Val Loss: {avg_val_loss:.4f} | "
                f"Detection Loss: {avg_train_detection_loss:.4f} | "
                f"Severity Loss: {avg_train_severity_loss:.4f} | "
                f"LR: {optimizer.param_groups[0]['lr']:.2e}"
            )
        
        # Early stopping (using validation loss; lower is better)
        if avg_val_loss < best_map - min_delta:
            best_map = avg_val_loss
            patience_counter = 0
            # Save EMA weights if enabled, else save current weights
            if use_ema and ema_state is not None:
                torch.save(ema_state, os.path.join(run_save_dir, "best_model.pth"))
                logger.info(f"New best EMA model saved with Val Loss: {best_map:.4f}")
            else:
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
        default=47,
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
    parser.add_argument(
        "--balanced_sampling", action="store_true", help="Enable class-balanced sampling"
    )
    parser.add_argument(
        "--use_ema", action="store_true", help="Use EMA weights during training and evaluation"
    )
    parser.add_argument(
        "--ema_decay", type=float, default=0.9998, help="EMA decay rate"
    )
    parser.add_argument(
        "--multiscale", action="store_true", help="Enable multi-scale training"
    )
    parser.add_argument(
        "--ms_min", type=int, default=480, help="Min short side for multi-scale"
    )
    parser.add_argument(
        "--ms_max", type=int, default=800, help="Max short side for multi-scale"
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
