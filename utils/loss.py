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

class YOLOXLoss(nn.Module):
    def __init__(self, num_classes, stride=16.0):
        super().__init__()
        self.num_classes = num_classes
        self.stride = stride

        self.bce_loss = nn.BCEWithLogitsLoss(reduction="none")
        # In YOLOX, the regression loss is often an IoU-based loss (like GIoU) or L1 loss on the decoded boxes.
        # We will use L1 loss here for simplicity and effectiveness, as it's a common choice.
        self.iou_loss = nn.L1Loss(reduction="none") 
        
        # SimOTA parameters
        self.center_sampling_radius = 2.5
        self.topk_candidates = 10
        self.reg_weight = 5.0 # Weight for the regression loss

    def forward(self, prediction_maps, targets):
        """
        Args:
            prediction_maps (tuple): (cls_pred, reg_pred, obj_pred) from the model
                - cls_pred: (B, C, H, W)
                - reg_pred: (B, 4, H, W)
                - obj_pred: (B, 1, H, W)
            targets (list of Tensors): List of [num_gt, 5] tensors -> [x1, y1, x2, y2, class_id]
        """
        cls_preds_map, reg_preds_map, obj_preds_map = prediction_maps
        device = cls_preds_map.device

        batch_size, _, H, W = cls_preds_map.shape
        num_predictions = H * W
        
        # 1. DECODE PREDICTIONS: Convert raw feature maps to a usable format
        # Flatten predictions from (B, C, H, W) to (B, H*W, C)
        cls_preds = cls_preds_map.permute(0, 2, 3, 1).reshape(batch_size, num_predictions, self.num_classes)
        reg_preds = reg_preds_map.permute(0, 2, 3, 1).reshape(batch_size, num_predictions, 4)
        obj_preds = obj_preds_map.permute(0, 2, 3, 1).reshape(batch_size, num_predictions, 1)

        # Generate grid coordinates needed for decoding
        yv, xv = torch.meshgrid([torch.arange(H), torch.arange(W)], indexing="ij")
        grid = torch.stack((xv, yv), 2).view(1, num_predictions, 2).to(device)
        
        # Decode regression predictions to (cx, cy, w, h) format
        decoded_reg_preds = torch.clone(reg_preds)
        decoded_reg_preds[..., :2] = (reg_preds[..., :2] + grid) * self.stride
        decoded_reg_preds[..., 2:] = torch.exp(reg_preds[..., 2:]) * self.stride
        
        total_cls_loss = 0.0
        total_reg_loss = 0.0
        total_obj_loss = 0.0
        num_fg = 0.0 # Total number of positive assignments (foreground)

        # 2. PERFORM LABEL ASSIGNMENT (SimOTA) for each image in the batch
        for b in range(batch_size):
            pred_cls_b = cls_preds[b]      # (H*W, C)
            pred_box_b = decoded_reg_preds[b] # (H*W, 4) in cxcywh format
            pred_obj_b = obj_preds[b]      # (H*W, 1)
            target_b = targets[b]          # (num_gt, 5) in xyxy format

            num_gt = target_b.shape[0]
            
            # If there are no ground truth objects, loss is only objectness loss for the background
            if num_gt == 0:
                obj_target = torch.zeros_like(pred_obj_b)
                loss_obj = self.bce_loss(pred_obj_b, obj_target).sum()
                total_obj_loss += loss_obj
                continue
            
            # 3. Get positive assignments using SimOTA
            fg_mask, assigned_gt_inds = self.get_assignments(
                pred_cls_b, pred_box_b, pred_obj_b, target_b
            )
            num_fg += fg_mask.sum()
            
            # 4. Prepare targets for the assigned positive predictions
            assigned_gts = target_b[assigned_gt_inds] # (num_fg, 5)
            
            # Create regression target (cxcywh format)
            gt_boxes_cxcywh = torch.stack((
                (assigned_gts[:, 0] + assigned_gts[:, 2]) / 2,
                (assigned_gts[:, 1] + assigned_gts[:, 3]) / 2,
                assigned_gts[:, 2] - assigned_gts[:, 0],
                assigned_gts[:, 3] - assigned_gts[:, 1],
            ), -1)
            
            cls_target = F.one_hot(assigned_gts[:, 4].long(), self.num_classes).float()
            obj_target = torch.zeros_like(pred_obj_b)
            obj_target[fg_mask] = 1.0 # Set objectness to 1 for positive assignments

            # 5. Calculate Losses
            loss_reg = self.iou_loss(pred_box_b[fg_mask], gt_boxes_cxcywh).sum()
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
    def get_assignments(self, pred_cls, pred_box_cxcywh, pred_obj, target):
        """SimOTA label assignment logic."""
        num_preds = pred_cls.shape[0]
        num_gt = target.shape[0]
        
        gt_boxes_xyxy = target[:, :4]
        gt_classes = target[:, 4]

        # Preliminary filtering
        gt_center = (gt_boxes_xyxy[:, :2] + gt_boxes_xyxy[:, 2:]) / 2
        pred_box_xyxy = cxcywh_to_xyxy(pred_box_cxcywh)
        
        is_in_box_and_center = self.get_in_box_info(pred_box_xyxy, gt_center, gt_boxes_xyxy)

        # Cost matrix calculation
        ious = self.calculate_iou(pred_box_xyxy, gt_boxes_xyxy)
        reg_cost = -torch.log(ious + 1e-8)
        
        pred_cls_sigmoid = torch.sigmoid(pred_cls)
        cls_cost = F.binary_cross_entropy(
            pred_cls_sigmoid.unsqueeze(1).repeat(1, num_gt, 1),
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
        
    def get_in_box_info(self, pred_boxes, gt_centers, gt_boxes):
        pred_centers = (pred_boxes[:, :2] + pred_boxes[:, 2:]) / 2
        
        x1, y1, x2, y2 = gt_boxes.unbind(-1)
        is_in_gts = (
            (pred_centers[:, 0].unsqueeze(1) > x1) & (pred_centers[:, 0].unsqueeze(1) < x2) &
            (pred_centers[:, 1].unsqueeze(1) > y1) & (pred_centers[:, 1].unsqueeze(1) < y2)
        )
        
        dist = torch.cdist(pred_centers, gt_centers)
        is_in_radius = dist < self.center_sampling_radius * self.stride
        
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