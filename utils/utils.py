# utils/utils.py
import torch
import numpy as np

def collate_fn(batch):
    """
    Puts each data field into a tensor with outer dimension batch size.
    Handles variable-sized bounding boxes.
    """
    images = []
    targets = []
    for image, target in batch:
        images.append(image)
        targets.append(target)
    images = torch.stack(images, dim=0)
    return images, targets


def box_iou(boxes1, boxes2):
    """
    Calculates the Intersection over Union (IoU) of two sets of bounding boxes.
    """
    # Implementation of box_iou (as provided earlier)
    # ... (Code for box_iou)
    pass # Replace with your box_iou implementation