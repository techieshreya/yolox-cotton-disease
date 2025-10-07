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
import xml.etree.ElementTree as ET
from collections import Counter

# Local imports
from utils.dataset import CottonDiseaseDataset, get_train_augs, get_val_augs
from utils.loss import YOLOXLoss
from models.yolox import YOLOX


def collate_fn(batch):
    images, targets = zip(*batch)
    images = torch.stack(images)
    targets = [
        torch.as_tensor(t, dtype=torch.float32)
        if not isinstance(t, torch.Tensor)
        else t.float()
        for t in targets
    ]
    return images, targets


def cxcywh_to_xyxy(boxes):
    """Convert center coordinates to corner coordinates (Tensor version)"""
    x_center, y_center, w, h = boxes.T
    x1 = x_center - w / 2
    y1 = y_center - h / 2
    x2 = x_center + w / 2
    y2 = y_center + h / 2
    return torch.stack([x1, y1, x2, y2], dim=1)


def postprocess(predictions, strides, num_classes, conf_thre=0.01, nms_thre=0.5):
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
        final_scores = obj_score * class_conf

        conf_mask = final_scores >= conf_thre

        # Convert boxes from [cx, cy, w, h] to [x1, y1, x2, y2]
        boxes_cxcywh = image_preds[:, :4][conf_mask]
        x_center, y_center, w, h = boxes_cxcywh.T
        x1 = x_center - w / 2
        y1 = y_center - h / 2
        x2 = x_center + w / 2
        y2 = y_center + h / 2
        boxes_xyxy = torch.stack([x1, y1, x2, y2], dim=1)

        detections = torch.cat(
            (
                boxes_xyxy,
                final_scores[conf_mask].unsqueeze(1),
                class_pred[conf_mask].unsqueeze(1).float(),
            ),
            1,
        )

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
            if (
                int(pred_classes[i]) == int(true_classes[best_gt_idx])
                and not gt_matched[best_gt_idx]
            ):
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
        if np.sum(recalls >= t) == 0:
            p = 0
        else:
            p = np.max(precisions[recalls >= t])
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


def calculate_map_per_class(
    pred_results, true_boxes, true_classes, num_classes, iou_threshold=0.5
):
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
        rect = patches.Rectangle(
            (box[0], box[1]),
            box[2] - box[0],
            box[3] - box[1],
            linewidth=2,
            edgecolor="r",
            facecolor="none",
        )
        ax.add_patch(rect)
        ax.text(
            box[0],
            box[1] - 5,
            f"{score:.2f}",
            color="white",
            bbox=dict(facecolor="red", alpha=0.5),
        )

    for box in gt_boxes:
        rect = patches.Rectangle(
            (box[0], box[1]),
            box[2] - box[0],
            box[3] - box[1],
            linewidth=2,
            edgecolor="g",
            facecolor="none",
        )
        ax.add_patch(rect)

    plt.title(f"Epoch {epoch} Predictions (Red) vs Ground Truth (Green)")
    plt.axis("off")
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f"predictions_epoch_{epoch}.png"))
    plt.close()


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create the augmentation pipelines
    train_augmentations = get_train_augs(args.input_size)
    val_augmentations = get_val_augs(args.input_size)

    # ---------- FIX 1: Corrected Dataset Initialization ----------
    train_dataset = CottonDiseaseDataset(
        data_dir=r"data/train",
        augmentations=train_augmentations,
        input_size=args.input_size,
    )
    val_dataset = CottonDiseaseDataset(
        data_dir=r"data/valid",
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

    # Model, Optimizer, Loss
    model = YOLOX(num_classes=args.num_classes, phi=args.phi).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=5e-4)

    # Add a learning rate warmup scheduler
    warmup_epochs = 5
    lr_warmup_factor = 1.0 / warmup_epochs
    main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs - warmup_epochs, eta_min=args.lr * 0.01
    )

    # Class weights to handle imbalance - give more weight to difficult classes
    # [curl_stage1, curl_stage2, healthy, leaf_enation, sooty]
    class_weights = [3.0, 1.0, 1.0, 1.0, 1.0]  # 3x weight for curl_stage1
    criterion = YOLOXLoss(
        num_classes=args.num_classes,
        strides=model.stride.tolist(),
        use_focal_loss=True,  # Enable focal loss for hard examples
        class_weights=class_weights,
    )

    best_map = 0.0
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0

        # LR Warmup
        if epoch < warmup_epochs:
            for i, param_group in enumerate(optimizer.param_groups):
                param_group["lr"] = args.lr * (epoch + 1) * lr_warmup_factor

        for images, targets_xyxy in tqdm(
            train_loader, desc=f"Train Epoch {epoch + 1}/{args.epochs}"
        ):
            images = images.to(device)
            targets_xyxy = [t.to(device) for t in targets_xyxy]

            optimizer.zero_grad()
            outputs = model(images)
            total_loss = criterion(outputs, targets_xyxy)
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += total_loss.item()

        # Validation phase
        model.eval()
        val_loss = 0.0
        all_aps = []
        # The new model's output is a list of tensors, one for each FPN level
        # We need a new way to store predictions for post-processing
        output_predictions = []

        start_time = time.time()
        with torch.no_grad():
            for batch_idx, (images, targets_xyxy) in enumerate(
                tqdm(val_loader, desc="Validating")
            ):
                images = images.to(device)
                targets_xyxy_device = [t.to(device) for t in targets_xyxy]

                outputs = model(images)
                val_loss_batch = criterion(outputs, targets_xyxy_device)
                val_loss += val_loss_batch.item()

                # Store raw outputs for post-processing after the loop
                # This gathers all predictions from all batches
                if batch_idx == 0:
                    for level_out in outputs:
                        output_predictions.append(level_out)
                else:
                    for i, level_out in enumerate(outputs):
                        output_predictions[i] = torch.cat(
                            (output_predictions[i], level_out), dim=0
                        )

            # Perform post-processing on all validation data at once
            final_detections = postprocess(
                output_predictions, model.stride.to(device), args.num_classes
            )

            per_class_aps = np.zeros(args.num_classes)
            for i in range(len(final_detections)):
                pred_results_np = final_detections[i].cpu().numpy()
                # We need to get the correct ground truth for each image
                # This assumes val_loader batch size is consistent, which is typical
                gt_idx = i
                gt_target = val_dataset[gt_idx][1].numpy()

                if gt_target.shape[0] > 0:
                    gt_boxes = gt_target[:, :4]
                    gt_classes = gt_target[:, 4]
                    aps = calculate_map_per_class(
                        pred_results_np, gt_boxes, gt_classes, args.num_classes
                    )
                    per_class_aps += np.array(aps)

            inference_time = time.time() - start_time
            per_class_aps /= len(final_detections)

            if epoch % 5 == 0:  # Visualize every 5 epochs
                # Get the first image of the validation set for visualization
                first_image, first_gt = val_dataset[0]
                visualize_predictions(
                    first_image,
                    final_detections[0].cpu().numpy(),
                    first_gt[:, :4].cpu().numpy(),
                    epoch,
                    args.save_dir,
                )

        mean_ap = np.mean(per_class_aps) if len(per_class_aps) > 0 else 0.0

        # Scheduler Step
        if epoch >= warmup_epochs:
            main_scheduler.step()

        # Print with color coding for problematic classes
        curl1_indicator = (
            "🔴" if per_class_aps[0] < 0.5 else "🟡" if per_class_aps[0] < 0.8 else "🟢"
        )
        print(
            f"Epoch {epoch + 1}/{args.epochs} | "
            f"Train Loss: {train_loss / len(train_loader):.4f} | "
            f"Val Loss: {val_loss / len(val_loader):.4f} | "
            f"mAP@0.5: {mean_ap:.4f} | "
            f"{curl1_indicator} AP (Curl stage-1): {per_class_aps[0]:.4f} | "
            f"AP (Curl stage-2): {per_class_aps[1]:.4f} | "
            f"AP (Healthy): {per_class_aps[2]:.4f} | "
            f"AP (Leaf Enation): {per_class_aps[3]:.4f} | "
            f"AP (Sooty): {per_class_aps[4]:.4f} | "
            f"Inference Time: {inference_time:.4f}s | "
            f"LR: {optimizer.param_groups[0]['lr']:.2e}"
        )

        if mean_ap > best_map:
            best_map = mean_ap
            torch.save(
                model.state_dict(), os.path.join(args.save_dir, "best_model.pth")
            )
            print(f"New best model saved with mAP: {best_map:.4f}")

        if (epoch + 1) % args.save_interval == 0:
            os.makedirs(args.save_dir, exist_ok=True)
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "map": mean_ap,
                },
                os.path.join(args.save_dir, f"epoch_{epoch + 1}.pth"),
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLOX Training Script")
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
        default=2,
        help="Batch size for training (reduced for YOLOX-M)",
    )
    parser.add_argument(
        "--epochs", type=int, default=200, help="Number of training epochs"
    )  # Increased epochs
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument(
        "--num_workers",
        type=int,
        default=47,
        help="Number of workers for data loading (0 for Windows is often safest)",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./checkpoints",
        help="Directory to save checkpoints",
    )
    parser.add_argument(
        "--save_interval", type=int, default=10, help="Save checkpoint every N epochs"
    )

    # This try-except block helps run it smoothly in some environments.
    try:
        args = parser.parse_args()
    except SystemExit:
        args = parser.parse_args([])

    os.makedirs(args.save_dir, exist_ok=True)
    train(args)
