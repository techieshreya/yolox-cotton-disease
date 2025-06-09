import torch
import numpy as np

def cxcywh_to_xyxy(boxes):
    """Convert center coordinates to corner coordinates"""
    if isinstance(boxes, torch.Tensor):
        x_center, y_center, w, h = boxes.unbind(-1)
        x1 = x_center - w/2
        y1 = y_center - h/2
        x2 = x_center + w/2
        y2 = y_center + h/2
        return torch.stack([x1, y1, x2, y2], dim=-1)
    else:  # numpy
        x_center, y_center, w, h = boxes.T
        x1 = x_center - w/2
        y1 = y_center - h/2
        x2 = x_center + w/2
        y2 = y_center + h/2
        return np.stack([x1, y1, x2, y2], axis=1)

def xyxy_to_cxcywh(boxes):
    """Convert corner coordinates to center coordinates"""
    if isinstance(boxes, torch.Tensor):
        x1, y1, x2, y2 = boxes.unbind(-1)
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        w = x2 - x1
        h = y2 - y1
        return torch.stack([cx, cy, w, h], dim=-1)
    else:  # numpy
        x1, y1, x2, y2 = boxes.T
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        w = x2 - x1
        h = y2 - y1
        return np.stack([cx, cy, w, h], axis=1)