import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

def plot_predictions(image, pred_boxes, pred_scores, gt_boxes, class_names, file_name):
    """Visualize predictions vs ground truth"""
    fig, ax = plt.subplots(1, figsize=(12, 8))
    
    # Unnormalize image if needed
    if isinstance(image, torch.Tensor):
        image = image.cpu().permute(1, 2, 0).numpy()
        image = (image * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406]))
        image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    # Draw predictions
    for box, score in zip(pred_boxes, pred_scores):
        rect = patches.Rectangle(
            (box[0], box[1]), box[2]-box[0], box[3]-box[1],
            linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        ax.text(box[0], box[1], f"{score:.2f}", 
                color='white', bbox=dict(facecolor='red', alpha=0.5))
    
    # Draw ground truth
    for box in gt_boxes:
        rect = patches.Rectangle(
            (box[0], box[1]), box[2]-box[0], box[3]-box[1],
            linewidth=2, edgecolor='g', facecolor='none')
        ax.add_patch(rect)
    
    plt.title("Predictions (Red) vs Ground Truth (Green)")
    plt.axis('off')
    plt.savefig(file_name)
    plt.close()