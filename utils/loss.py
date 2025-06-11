import torch
import torch.nn as nn
import torch.nn.functional as F

def cxcywh_to_xyxy(boxes):
    """Convert center coordinates to corner coordinates (Tensor version)"""
    cx, cy, w, h = boxes.unbind(-1)
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    return torch.stack((x1, y1, x2, y2), dim=-1)

def giou_loss(pred_boxes_xyxy, gt_boxes_xyxy, reduction="none"):
    """
    Calculate GIoU loss.
    Args:
        pred_boxes_xyxy (Tensor): Predicted boxes, shape (N, 4), format (x1, y1, x2, y2)
        gt_boxes_xyxy (Tensor): Ground truth boxes, shape (N, 4), format (x1, y1, x2, y2)
    Returns:
        Tensor: GIoU loss
    """
    # Intersection
    inter_x1 = torch.max(pred_boxes_xyxy[:, 0], gt_boxes_xyxy[:, 0])
    inter_y1 = torch.max(pred_boxes_xyxy[:, 1], gt_boxes_xyxy[:, 1])
    inter_x2 = torch.min(pred_boxes_xyxy[:, 2], gt_boxes_xyxy[:, 2])
    inter_y2 = torch.min(pred_boxes_xyxy[:, 3], gt_boxes_xyxy[:, 3])
    inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)

    # Union
    pred_area = (pred_boxes_xyxy[:, 2] - pred_boxes_xyxy[:, 0]) * (pred_boxes_xyxy[:, 3] - pred_boxes_xyxy[:, 1])
    gt_area = (gt_boxes_xyxy[:, 2] - gt_boxes_xyxy[:, 0]) * (gt_boxes_xyxy[:, 3] - gt_boxes_xyxy[:, 1])
    union_area = pred_area + gt_area - inter_area
    
    # IoU
    iou = inter_area / (union_area + 1e-6)

    # Bounding box of the union
    c_x1 = torch.min(pred_boxes_xyxy[:, 0], gt_boxes_xyxy[:, 0])
    c_y1 = torch.min(pred_boxes_xyxy[:, 1], gt_boxes_xyxy[:, 1])
    c_x2 = torch.max(pred_boxes_xyxy[:, 2], gt_boxes_xyxy[:, 2])
    c_y2 = torch.max(pred_boxes_xyxy[:, 3], gt_boxes_xyxy[:, 3])
    c_area = (c_x2 - c_x1) * (c_y2 - c_y1)
    
    giou = iou - (c_area - union_area) / (c_area + 1e-6)
    
    loss = 1.0 - giou
    
    if reduction == "sum":
        return loss.sum()
    elif reduction == "mean":
        return loss.mean()
    else: # "none"
        return loss

class YOLOXLoss(nn.Module):
    def __init__(self, num_classes, strides=[8, 16, 32]):
        super().__init__()
        self.num_classes = num_classes
        self.strides = strides

        self.bce_loss = nn.BCEWithLogitsLoss(reduction="none")
        # In YOLOX, the regression loss is often an IoU-based loss (like GIoU) or L1 loss on the decoded boxes.
        # We will use GIoU loss, which is generally better for object detection.
        self.iou_loss = giou_loss
        
        # SimOTA parameters
        self.center_sampling_radius = 2.5
        self.topk_candidates = 10
        self.reg_weight = 5.0 # Weight for the regression loss

    def forward(self, fpn_outputs, targets):
        """
        Args:
            fpn_outputs (list of Tensors): Output from the YOLOXHead, a list of [B, 5+C, H, W] tensors.
            targets (list of Tensors): List of [num_gt, 5] tensors -> [x1, y1, x2, y2, class_id]
        """
        device = targets[0].device
        batch_size = fpn_outputs[0].shape[0]

        # 1. DECODE PREDICTIONS FROM ALL FPN LEVELS
        all_cls_preds, all_reg_preds, all_obj_preds, all_strides = [], [], [], []

        for i, fpn_out in enumerate(fpn_outputs):
            stride = self.strides[i]
            B, _, H, W = fpn_out.shape
            
            # Generate grid and reshape raw output
            yv, xv = torch.meshgrid([torch.arange(H), torch.arange(W)], indexing="ij")
            grid = torch.stack((xv, yv), 2).view(1, -1, 2).to(device)
            
            fpn_out = fpn_out.permute(0, 2, 3, 1).reshape(B, -1, 5 + self.num_classes)
            
            # Split into regression, objectness, and classification predictions
            reg_preds = fpn_out[..., :4]
            obj_preds = fpn_out[..., 4:5]
            cls_preds = fpn_out[..., 5:]
            
            # Decode regression predictions
            decoded_reg_preds = torch.clone(reg_preds)
            decoded_reg_preds[..., :2] = (reg_preds[..., :2] + grid) * stride
            decoded_reg_preds[..., 2:] = torch.exp(reg_preds[..., 2:]) * stride
            
            all_reg_preds.append(decoded_reg_preds)
            all_obj_preds.append(obj_preds)
            all_cls_preds.append(cls_preds)
            all_strides.append(torch.full((B, grid.shape[1], 1), stride, device=device))

        # Concatenate predictions from all levels
        cat_reg_preds = torch.cat(all_reg_preds, dim=1)
        cat_obj_preds = torch.cat(all_obj_preds, dim=1)
        cat_cls_preds = torch.cat(all_cls_preds, dim=1)
        cat_strides = torch.cat(all_strides, dim=1)
        
        total_cls_loss = 0.0
        total_reg_loss = 0.0
        total_obj_loss = 0.0
        num_fg = 0.0 # Total number of positive assignments

        # 2. PERFORM LABEL ASSIGNMENT (SimOTA) for each image
        for b in range(batch_size):
            pred_cls_b = cat_cls_preds[b]
            pred_box_b = cat_reg_preds[b]
            pred_obj_b = cat_obj_preds[b]
            strides_b = cat_strides[b]
            target_b = targets[b]

            num_gt = target_b.shape[0]
            
            if num_gt == 0:
                obj_target = torch.zeros_like(pred_obj_b)
                loss_obj = self.bce_loss(pred_obj_b, obj_target).sum()
                total_obj_loss += loss_obj
                continue
            
            # 3. Get positive assignments using SimOTA
            fg_mask, assigned_gt_inds = self.get_assignments(
                pred_cls_b, pred_box_b, pred_obj_b, target_b, strides_b
            )
            num_fg += fg_mask.sum()
            
            # 4. Prepare targets
            assigned_gts = target_b[assigned_gt_inds]
            
            gt_boxes_cxcywh = torch.stack((
                (assigned_gts[:, 0] + assigned_gts[:, 2]) / 2,
                (assigned_gts[:, 1] + assigned_gts[:, 3]) / 2,
                assigned_gts[:, 2] - assigned_gts[:, 0],
                assigned_gts[:, 3] - assigned_gts[:, 1],
            ), -1)
            
            cls_target = F.one_hot(assigned_gts[:, 4].long(), self.num_classes).float()
            obj_target = torch.zeros_like(pred_obj_b)
            obj_target[fg_mask] = 1.0 # Objectness is 1 for positive predictions

            # 5. Calculate Losses
            pred_xyxy = cxcywh_to_xyxy(pred_box_b[fg_mask])
            gt_xyxy = cxcywh_to_xyxy(gt_boxes_cxcywh)

            loss_reg = self.iou_loss(pred_xyxy, gt_xyxy, reduction="sum")
            loss_cls = self.bce_loss(pred_cls_b[fg_mask], cls_target).sum()
            loss_obj = self.bce_loss(pred_obj_b, obj_target).sum()
            
            total_reg_loss += loss_reg
            total_cls_loss += loss_cls
            total_obj_loss += loss_obj

        # Normalize losses
        num_fg = max(num_fg, 1)
        total_loss = (self.reg_weight * total_reg_loss + total_cls_loss + total_obj_loss) / num_fg
        
        return total_loss

    @torch.no_grad()
    def get_assignments(self, pred_cls, pred_box_cxcywh, pred_obj, target, strides_tensor):
        """SimOTA for multi-level predictions."""
        num_preds = pred_cls.shape[0]
        num_gt = target.shape[0]
        
        gt_boxes_xyxy = target[:, :4]
        gt_classes = target[:, 4]

        # Preliminary filtering using box centers
        gt_center = (gt_boxes_xyxy[:, :2] + gt_boxes_xyxy[:, 2:]) / 2
        pred_box_xyxy = cxcywh_to_xyxy(pred_box_cxcywh)
        
        is_in_box_and_center = self.get_in_box_info(pred_box_xyxy, gt_center, gt_boxes_xyxy, strides_tensor)

        # Cost matrix calculation
        ious = self.calculate_iou(pred_box_xyxy, gt_boxes_xyxy)
        reg_cost = -torch.log(ious + 1e-8)
        
        cls_cost = F.binary_cross_entropy(
            torch.sigmoid(pred_cls).unsqueeze(1).repeat(1, num_gt, 1),
            F.one_hot(gt_classes.long(), self.num_classes).float().unsqueeze(0).repeat(num_preds, 1, 1),
            reduction="none"
        ).sum(-1)

        cost_matrix = 3.0 * reg_cost + 1.0 * cls_cost
        cost_matrix[~is_in_box_and_center] = float("inf")

        # Dynamic K estimation
        topk_ious, _ = torch.topk(ious, self.topk_candidates, dim=0)
        dynamic_k = torch.clamp(topk_ious.sum(0).int(), min=1)

        # Final assignment
        fg_mask = torch.zeros(num_preds, dtype=torch.bool, device=pred_cls.device)
        assigned_gt_inds = torch.zeros(num_preds, dtype=torch.long, device=pred_cls.device)
        
        for gt_idx in range(num_gt):
            k = dynamic_k[gt_idx]
            _, topk_inds = torch.topk(cost_matrix[:, gt_idx], k, largest=False)
            fg_mask[topk_inds] = True
            assigned_gt_inds[topk_inds] = gt_idx
        
        return fg_mask, assigned_gt_inds[fg_mask]
        
    def get_in_box_info(self, pred_boxes, gt_centers, gt_boxes, strides_tensor):
        pred_centers = (pred_boxes[:, :2] + pred_boxes[:, 2:]) / 2

        x1, y1, x2, y2 = gt_boxes.unbind(-1)
        is_in_gts = (
            (pred_centers[:, 0].unsqueeze(1) > x1) & (pred_centers[:, 0].unsqueeze(1) < x2) &
            (pred_centers[:, 1].unsqueeze(1) > y1) & (pred_centers[:, 1].unsqueeze(1) < y2)
        )
        
        dist = torch.cdist(pred_centers, gt_centers)
        is_in_radius = dist < self.center_sampling_radius * strides_tensor
        
        return is_in_gts & is_in_radius

    def calculate_iou(self, boxes1_xyxy, boxes2_xyxy):
        boxes1 = boxes1_xyxy.unsqueeze(1)
        boxes2 = boxes2_xyxy.unsqueeze(0)
        inter_x1 = torch.max(boxes1[..., 0], boxes2[..., 0])
        inter_y1 = torch.max(boxes1[..., 1], boxes2[..., 1])
        inter_x2 = torch.min(boxes1[..., 2], boxes2[..., 2])
        inter_y2 = torch.min(boxes1[..., 3], boxes2[..., 3])
        intersection = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)
        area1 = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
        area2 = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])
        union = area1 + area2 - intersection
        return intersection / (union + 1e-6)