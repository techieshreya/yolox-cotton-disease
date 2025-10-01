


import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision import ops
from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import time
import psutil
from torch.cuda.amp import GradScaler, autocast

# Local imports
from utils.dataset import CottonDiseaseDataset, get_train_augs, get_val_augs
from utils.loss import YOLOXLoss
from models.yolox import YOLOX


def get_gpu_memory():
    """Get available GPU memory in GB"""
    if torch.cuda.is_available():
        gpu_props = torch.cuda.get_device_properties(0)
        total_memory = gpu_props.total_memory / (1024**3)  # Convert to GB
        torch.cuda.empty_cache()
        allocated = torch.cuda.memory_allocated(0) / (1024**3)
        reserved = torch.cuda.memory_reserved(0) / (1024**3)
        free_memory = total_memory - reserved
        return total_memory, free_memory
    return 0, 0


def get_cpu_memory():
    """Get available CPU memory in GB"""
    mem = psutil.virtual_memory()
    return mem.total / (1024**3), mem.available / (1024**3)


def estimate_model_memory(phi, input_size, num_classes):
    """Estimate model memory requirement in GB"""
    base_params = {
        's': 9.0,   # ~9M parameters
        'm': 25.3,  # ~25M parameters
        'l': 54.2,  # ~54M parameters
        'x': 99.1   # ~99M parameters
    }

    # Rough estimation: 4 bytes per parameter (float32)
    # Plus activations (roughly 2-3x model size during training)
    params_mb = base_params.get(phi, 9.0)
    model_memory = (params_mb * 4 / 1000) * 3.5  # Convert to GB with activation overhead

    # Add memory for input batch (rough estimate)
    input_memory_per_image = (3 * input_size[0] * input_size[1] * 4) / (1024**3)

    return model_memory, input_memory_per_image


def optimize_parameters(total_vram, free_vram, num_classes=5):
    """
    Automatically determine optimal training parameters based on available VRAM

    Returns:
        dict: Optimized parameters
    """
    print(f"\n{'='*60}")
    print(f"🚀 AUTO-OPTIMIZATION FOR MAX PERFORMANCE")
    print(f"{'='*60}")
    print(f"Total VRAM: {total_vram:.2f} GB")
    print(f"Available VRAM: {free_vram:.2f} GB")

    # Define parameter sets for different VRAM levels
    # Format: (phi, input_size, base_batch_size, use_amp)
    vram_configs = [
        (24, 'x', [640, 640], 16, True),   # 24GB+ (RTX 3090/4090, A5000, etc.)
        (16, 'l', [640, 640], 12, True),   # 16GB (RTX 4080, A4000, etc.)
        (12, 'l', [640, 640], 8, True),    # 12GB (RTX 3080 Ti, 4070 Ti)
        (10, 'm', [640, 640], 8, True),    # 10GB (RTX 3080)
        (8, 'm', [640, 640], 6, True),     # 8GB (RTX 3070, 4060 Ti)
        (6, 's', [640, 640], 4, True),     # 6GB (RTX 3060, 2060)
        (4, 's', [512, 512], 2, True),     # 4GB (GTX 1650, etc.)
        (0, 's', [416, 416], 1, True),     # < 4GB
    ]

    # Select configuration based on free VRAM
    config = None
    for vram_threshold, phi, input_size, batch_size, use_amp in vram_configs:
        if free_vram >= vram_threshold:
            config = (phi, input_size, batch_size, use_amp)
            break

    if config is None:
        config = ('s', [416, 416], 1, True)

    phi, input_size, batch_size, use_amp = config

    # Estimate actual memory usage
    model_mem, input_mem_per_img = estimate_model_memory(phi, input_size, num_classes)

    # Fine-tune batch size to maximize VRAM usage (leave 2GB buffer)
    target_memory = free_vram - 2.0
    estimated_batch_memory = model_mem + (batch_size * input_mem_per_img * 4)  # 4x for gradients/optimizer

    if estimated_batch_memory < target_memory:
        # Can increase batch size
        max_batch_size = int((target_memory - model_mem) / (input_mem_per_img * 4))
        batch_size = min(max_batch_size, batch_size + 8)
    elif estimated_batch_memory > target_memory:
        # Need to decrease batch size
        max_batch_size = max(1, int((target_memory - model_mem) / (input_mem_per_img * 4)))
        batch_size = max_batch_size

    # Determine number of workers based on CPU
    cpu_total, cpu_available = get_cpu_memory()
    cpu_count = os.cpu_count() or 4
    # Use 75% of available CPU cores, max 8
    num_workers = min(max(1, int(cpu_count * 0.75)), 8)

    # Adjust learning rate based on batch size (linear scaling rule)
    base_lr = 1e-3
    lr = base_lr * (batch_size / 8)  # Scale LR with batch size

    # Gradient accumulation for small batch sizes
    accumulation_steps = 1
    effective_batch_size = batch_size
    if batch_size < 4:
        accumulation_steps = max(1, 8 // batch_size)
        effective_batch_size = batch_size * accumulation_steps
        print(f"⚠️  Small batch size detected. Using gradient accumulation: {accumulation_steps} steps")

    # Determine optimal settings
    params = {
        'phi': phi,
        'input_size': input_size,
        'batch_size': batch_size,
        'effective_batch_size': effective_batch_size,
        'accumulation_steps': accumulation_steps,
        'num_workers': num_workers,
        'lr': lr,
        'use_amp': use_amp,
        'pin_memory': True,
        'persistent_workers': num_workers > 0,
        'prefetch_factor': 2 if num_workers > 0 else None,
    }

    print(f"\n✅ OPTIMIZED CONFIGURATION:")
    print(f"{'─'*60}")
    print(f"Model Size (phi):          {phi.upper()}")
    print(f"Input Size:                {input_size[0]}x{input_size[1]}")
    print(f"Batch Size:                {batch_size}")
    print(f"Effective Batch Size:      {effective_batch_size}")
    print(f"Gradient Accumulation:     {accumulation_steps} steps")
    print(f"Learning Rate:             {lr:.2e}")
    print(f"Mixed Precision (AMP):     {use_amp}")
    print(f"Num Workers:               {num_workers}")
    print(f"Pin Memory:                {params['pin_memory']}")
    print(f"Persistent Workers:        {params['persistent_workers']}")
    print(f"Estimated Model Memory:    {model_mem:.2f} GB")
    print(f"Estimated Total Memory:    {model_mem + batch_size * input_mem_per_img * 4:.2f} GB")
    print(f"{'─'*60}\n")

    return params


def collate_fn(batch):
    images, targets = zip(*batch)
    images = torch.stack(images)
    targets = [torch.as_tensor(t, dtype=torch.float32) if not isinstance(t, torch.Tensor) else t.float() for t in targets]
    return images, targets


def cxcywh_to_xyxy(boxes):
    """Convert center coordinates to corner coordinates (Tensor version)"""
    x_center, y_center, w, h = boxes.T
    x1 = x_center - w / 2
    y1 = y_center - h / 2
    x2 = x_center + w / 2
    y2 = y_center + h / 2
    return torch.stack([x1, y1, x2, y2], dim=1)


def postprocess(predictions, strides, num_classes, conf_thre=0.05, nms_thre=0.5):
    """Post-processes raw predictions from the model."""
    all_detections = []
    device = predictions[0].device

    # 1. Decode predictions from all levels
    decoded_preds = []
    for i, preds_level in enumerate(predictions):
        stride = strides[i]
        N, C, H, W = preds_level.shape

        # Create grid
        yv, xv = torch.meshgrid([torch.arange(H, device=device), torch.arange(W, device=device)], indexing="ij")
        grid = torch.stack((xv, yv), 2).view(1, H*W, 2).repeat(N, 1, 1)

        # Reshape and decode
        preds_level = preds_level.permute(0, 2, 3, 1).reshape(N, H*W, C)

        box_xy = (preds_level[..., :2] + grid) * stride
        box_wh = torch.exp(preds_level[..., 2:4]) * stride

        # Combine decoded boxes with obj and cls scores
        decoded_level = torch.cat((box_xy, box_wh, preds_level[..., 4:]), dim=-1)
        decoded_preds.append(decoded_level)

    output = torch.cat(decoded_preds, dim=1)

    # 2. Loop through images in the batch to perform NMS
    for i in range(output.shape[0]):
        image_preds = output[i]

        # Apply score threshold
        obj_score = image_preds[:, 4].sigmoid()
        cls_scores = image_preds[:, 5:].sigmoid()

        class_conf, class_pred = torch.max(cls_scores, 1)
        final_scores = obj_score * class_conf

        conf_mask = (final_scores >= conf_thre)

        # Convert boxes from [cx, cy, w, h] to [x1, y1, x2, y2]
        boxes_cxcywh = image_preds[:, :4][conf_mask]
        x_center, y_center, w, h = boxes_cxcywh.T
        x1 = x_center - w / 2
        y1 = y_center - h / 2
        x2 = x_center + w / 2
        y2 = y_center + h / 2
        boxes_xyxy = torch.stack([x1, y1, x2, y2], dim=1)

        detections = torch.cat((
            boxes_xyxy,
            final_scores[conf_mask].unsqueeze(1),
            class_pred[conf_mask].unsqueeze(1).float()
        ), 1)

        if not detections.shape[0]:
            all_detections.append(torch.empty((0, 6), device=device))
            continue

        # Perform NMS
        nms_out_index = ops.nms(detections[:, :4], detections[:, 4], nms_thre)
        all_detections.append(detections[nms_out_index])

    return all_detections


def calculate_map(pred_results, true_boxes, true_classes, iou_threshold=0.5):
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

        if ious[best_gt_idx] > iou_threshold:
            if int(pred_classes[i]) == int(true_classes[best_gt_idx]) and not gt_matched[best_gt_idx]:
                tp[i] = 1
                gt_matched[best_gt_idx] = 1
            else:
                fp[i] = 1
        else:
            fp[i] = 1

    tp_cumsum = np.cumsum(tp)
    fp_cumsum = np.cumsum(fp)
    recalls = tp_cumsum / (len(true_boxes) + 1e-6)
    precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-6)

    ap = 0.0
    for t in np.arange(0, 1.1, 0.1):
        if np.sum(recalls >= t) == 0: p = 0
        else: p = np.max(precisions[recalls >= t])
        ap += p / 11.0
    return ap


def calculate_iou(box1, boxes):
    x1 = np.maximum(box1[0], boxes[:, 0])
    y1 = np.maximum(box1[1], boxes[:, 1])
    x2 = np.minimum(box1[2], boxes[:, 2])
    y2 = np.minimum(box1[3], boxes[:, 3])
    intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union = area1 + area2 - intersection
    return intersection / (union + 1e-6)


def calculate_map_per_class(pred_results, true_boxes, true_classes, num_classes, iou_threshold=0.5):
    aps = []
    for cls in range(num_classes):
        pred_mask = pred_results[:, 5] == cls
        gt_mask = true_classes == cls
        ap = calculate_map(
            pred_results[pred_mask],
            true_boxes[gt_mask],
            true_classes[gt_mask],
            iou_threshold
        )
        aps.append(ap)
    return aps


def visualize_predictions(image, pred_boxes, gt_boxes, epoch, save_dir):
    fig, ax = plt.subplots(1, figsize=(12, 8))
    image = image.cpu().permute(1, 2, 0).numpy()
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    image = np.clip(image, 0, 1)
    ax.imshow(image)

    for box_info in pred_boxes:
        box, score = box_info[:4], box_info[4]
        rect = patches.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        ax.text(box[0], box[1] - 5, f"{score:.2f}", color='white', bbox=dict(facecolor='red', alpha=0.5))

    for box in gt_boxes:
        rect = patches.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], linewidth=2, edgecolor='g', facecolor='none')
        ax.add_patch(rect)

    plt.title(f"Epoch {epoch} Predictions (Red) vs Ground Truth (Green)")
    plt.axis('off')
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f"predictions_epoch_{epoch}.png"))
    plt.close()


def train(args):
    # Detect hardware and optimize parameters
    if torch.cuda.is_available():
        device = torch.device('cuda')
        total_vram, free_vram = get_gpu_memory()
        optimal_params = optimize_parameters(total_vram, free_vram, args.num_classes)
    else:
        print("⚠️  No GPU detected! Using CPU (not recommended for training)")
        device = torch.device('cpu')
        optimal_params = {
            'phi': 's',
            'input_size': [416, 416],
            'batch_size': 1,
            'effective_batch_size': 1,
            'accumulation_steps': 1,
            'num_workers': 2,
            'lr': 1e-4,
            'use_amp': False,
            'pin_memory': False,
            'persistent_workers': False,
            'prefetch_factor': None,
        }

    # Override with optimal parameters unless explicitly set
    phi = args.phi if args.phi != 'auto' else optimal_params['phi']
    input_size = args.input_size if args.input_size != [0, 0] else optimal_params['input_size']
    batch_size = args.batch_size if args.batch_size != 0 else optimal_params['batch_size']
    num_workers = args.num_workers if args.num_workers != -1 else optimal_params['num_workers']
    lr = args.lr if args.lr != 0.0 else optimal_params['lr']
    use_amp = optimal_params['use_amp'] and torch.cuda.is_available()
    accumulation_steps = optimal_params['accumulation_steps']

    print(f"\n🎯 Starting training with device: {device}")

    # Create the augmentation pipelines
    train_augmentations = get_train_augs(input_size)
    val_augmentations = get_val_augs(input_size)

    # Datasets
    train_dataset = CottonDiseaseDataset(
        data_dir="data/train",
        augmentations=train_augmentations,
        input_size=input_size
    )
    val_dataset = CottonDiseaseDataset(
        data_dir="data/valid",
        augmentations=val_augmentations,
        input_size=input_size
    )

    print(f"📊 Dataset: {len(train_dataset)} train, {len(val_dataset)} val images")

    # DataLoaders with optimized settings
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=optimal_params['pin_memory'],
        persistent_workers=optimal_params['persistent_workers'],
        prefetch_factor=optimal_params['prefetch_factor']
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=optimal_params['pin_memory'],
        persistent_workers=optimal_params['persistent_workers'],
        prefetch_factor=optimal_params['prefetch_factor']
    )

    # Model
    model = YOLOX(num_classes=args.num_classes, phi=phi).to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"🔧 Model: YOLOX-{phi.upper()} | Params: {total_params/1e6:.2f}M (Trainable: {trainable_params/1e6:.2f}M)")

    # Optimizer with optimal settings
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=5e-4,
        betas=(0.9, 0.999)
    )

    # Learning rate scheduler with warmup
    warmup_epochs = 5
    lr_warmup_factor = 1.0 / warmup_epochs
    main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs - warmup_epochs, eta_min=lr * 0.01
    )

    # Loss function
    criterion = YOLOXLoss(num_classes=args.num_classes, strides=model.stride.tolist())

    # Mixed precision scaler
    scaler = GradScaler() if use_amp else None

    if use_amp:
        print("⚡ Mixed Precision Training: ENABLED")

    print(f"\n{'='*60}")
    print("🚀 STARTING TRAINING")
    print(f"{'='*60}\n")

    best_map = 0.0
    start_time_total = time.time()
    amp_failed = False

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        epoch_start = time.time()

        # LR Warmup
        if epoch < warmup_epochs:
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr * (epoch + 1) * lr_warmup_factor

        # Training loop
        optimizer.zero_grad()
        for batch_idx, (images, targets_xyxy) in enumerate(tqdm(train_loader, desc=f'Epoch {epoch+1}/{args.epochs}')):
            images = images.to(device, non_blocking=True)
            targets_xyxy = [t.to(device, non_blocking=True) for t in targets_xyxy]

            try:
                # Forward pass (model handles its own precision)
                outputs = model(images)

                # Loss computation (criterion not compatible with autocast)
                total_loss = criterion(outputs, targets_xyxy)
                total_loss = total_loss / accumulation_steps

                if use_amp and not amp_failed:
                    scaler.scale(total_loss).backward()

                else:
                    total_loss.backward()

                # Gradient accumulation
                if (batch_idx + 1) % accumulation_steps == 0:
                    if use_amp and not amp_failed:
                        scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    if use_amp and not amp_failed:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    optimizer.zero_grad()

                train_loss += total_loss.item() * accumulation_steps

            except RuntimeError as e:
                if "type" in str(e).lower() and "should be the same" in str(e).lower():
                    if use_amp and not amp_failed:
                        print("\n⚠️  Mixed precision incompatibility detected. Disabling AMP and continuing...")
                        amp_failed = True
                        use_amp = False
                        optimizer.zero_grad()
                        continue
                else:
                    raise

        # Handle remaining gradients
        if len(train_loader) % accumulation_steps != 0:
            if use_amp:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            optimizer.zero_grad()

        avg_train_loss = train_loss / len(train_loader)

        # Validation phase
        model.eval()
        val_loss = 0.0
        output_predictions = []

        val_start = time.time()
        with torch.no_grad():
            for batch_idx, (images, targets_xyxy) in enumerate(tqdm(val_loader, desc='Validating')):
                images = images.to(device, non_blocking=True)
                targets_xyxy_device = [t.to(device, non_blocking=True) for t in targets_xyxy]

                # Forward pass (model handles its own precision)
                outputs = model(images)

                # Loss computation (criterion not compatible with autocast)
                val_loss_batch = criterion(outputs, targets_xyxy_device)

                val_loss += val_loss_batch.item()

                # Store raw outputs for post-processing
                if batch_idx == 0:
                    for level_out in outputs:
                        output_predictions.append(level_out)
                else:
                    for i, level_out in enumerate(outputs):
                        output_predictions[i] = torch.cat((output_predictions[i], level_out), dim=0)

            # Perform post-processing
            final_detections = postprocess(output_predictions, model.stride.to(device), args.num_classes)

            per_class_aps = np.zeros(args.num_classes)
            for i in range(len(final_detections)):
                pred_results_np = final_detections[i].cpu().numpy()
                gt_idx = i
                gt_target = val_dataset[gt_idx][1].numpy()

                if gt_target.shape[0] > 0:
                    gt_boxes = gt_target[:, :4]
                    gt_classes = gt_target[:, 4]
                    aps = calculate_map_per_class(pred_results_np, gt_boxes, gt_classes, args.num_classes)
                    per_class_aps += np.array(aps)

            val_time = time.time() - val_start
            per_class_aps /= max(len(final_detections), 1)

        # Visualization (outside no_grad for matplotlib)
        mean_ap = np.mean(per_class_aps) if len(per_class_aps) > 0 else 0.0

        if epoch % 5 == 0:
            try:
                first_image, first_gt = val_dataset[0]
                visualize_predictions(
                    first_image,
                    final_detections[0].cpu().numpy(),
                    first_gt[:, :4].cpu().numpy(),
                    epoch,
                    args.save_dir
                )
                print(f'📊 Visualization saved: predictions_epoch_{epoch}.png')
            except Exception as e:
                print(f'⚠️  Could not save visualization: {e}')
        avg_val_loss = val_loss / len(val_loader)
        epoch_time = time.time() - epoch_start

        # Scheduler step
        if epoch >= warmup_epochs:
            main_scheduler.step()

        # Memory stats
        if torch.cuda.is_available():
            allocated_mem = torch.cuda.memory_allocated(0) / (1024**3)
            max_mem = torch.cuda.max_memory_allocated(0) / (1024**3)
            mem_str = f"| GPU Mem: {allocated_mem:.2f}/{max_mem:.2f} GB"
        else:
            mem_str = ""

        print(f'\n📈 Epoch {epoch+1}/{args.epochs} Summary:')
        print(f'├─ Train Loss: {avg_train_loss:.4f}')
        print(f'├─ Val Loss:   {avg_val_loss:.4f}')
        print(f'├─ mAP@0.5:    {mean_ap:.4f}')
        print(f'├─ Class APs:  {" | ".join([f"{ap:.3f}" for ap in per_class_aps])}')
        print(f'├─ Epoch Time: {epoch_time:.2f}s (Val: {val_time:.2f}s)')
        print(f'├─ LR:         {optimizer.param_groups[0]["lr"]:.2e}')
        print(f'└─ {mem_str}' if mem_str else '')

        # Save best model
        if mean_ap > best_map:
            best_map = mean_ap
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'map': mean_ap,
                'config': {
                    'phi': phi,
                    'input_size': input_size,
                    'batch_size': batch_size,
                    'num_classes': args.num_classes
                }
            }, os.path.join(args.save_dir, 'best_model.pth'))
            print(f'💾 New best model saved! mAP: {best_map:.4f}')

        # Save checkpoint
        if (epoch + 1) % args.save_interval == 0:
            os.makedirs(args.save_dir, exist_ok=True)
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'map': mean_ap,
            }, os.path.join(args.save_dir, f'epoch_{epoch+1}.pth'))
            print(f'💾 Checkpoint saved: epoch_{epoch+1}.pth')

        # Clear CUDA cache periodically
        if torch.cuda.is_available() and (epoch + 1) % 10 == 0:
            torch.cuda.empty_cache()

    total_time = time.time() - start_time_total
    print(f"\n{'='*60}")
    print(f"✅ TRAINING COMPLETE!")
    print(f"{'='*60}")
    print(f"Total Time: {total_time/3600:.2f} hours")
    print(f"Best mAP@0.5: {best_map:.4f}")
    print(f"Model saved to: {args.save_dir}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='YOLOX MAX Performance Training Script')
    parser.add_argument('--input_size', type=int, nargs=2, default=[0, 0], help='Model input size [height, width] (0 0 for auto)')
    parser.add_argument('--num_classes', type=int, default=5, help='Number of object classes')
    parser.add_argument('--phi', type=str, default='auto', help="Model size: 's', 'm', 'l', 'x', or 'auto'")
    parser.add_argument('--batch_size', type=int, default=0, help='Batch size for training (0 for auto)')
    parser.add_argument('--epochs', type=int, default=150, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.0, help='Learning rate (0.0 for auto)')
    parser.add_argument('--num_workers', type=int, default=-1, help='Number of workers for data loading (-1 for auto)')
    parser.add_argument('--save_dir', type=str, default='./checkpoints', help='Directory to save checkpoints')
    parser.add_argument('--save_interval', type=int, default=10, help='Save checkpoint every N epochs')

    try:
        args = parser.parse_args()
    except SystemExit:
        args = parser.parse_args([])

    os.makedirs(args.save_dir, exist_ok=True)

    # Print system info
    print(f"\n{'='*60}")
    print("🖥️  SYSTEM INFORMATION")
    print(f"{'='*60}")
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"GPU Device: {torch.cuda.get_device_name(0)}")
        print(f"GPU Count: {torch.cuda.device_count()}")

    cpu_total, cpu_available = get_cpu_memory()
    print(f"CPU Cores: {os.cpu_count()}")
    print(f"CPU Memory: {cpu_total:.2f} GB (Available: {cpu_available:.2f} GB)")
    print(f"{'='*60}\n")

    train(args)
