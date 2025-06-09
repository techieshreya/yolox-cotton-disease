# utils/transforms.py
import albumentations as A
from albumentations.pytorch import ToTensorV2
import config  # Import your config.py

def get_train_transforms():
    transforms = [
        A.Resize(640, 640),  # YOLOX input size (adjust if needed)
        A.HorizontalFlip(p=config.AUGMENTATIONS['flip_prob']),
        A.RandomRotate90(p=config.AUGMENTATIONS['rotate_prob']),
        A.RandomBrightnessContrast(
            p=config.AUGMENTATIONS['brightness_contrast_prob'],
            brightness_limit=config.AUGMENTATIONS['brightness_limit'],
            contrast_limit=config.AUGMENTATIONS['contrast_limit'],
        ),
        ToTensorV2(p=1.0),  # Convert to PyTorch tensor
    ]
    return A.Compose(transforms, bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))


def get_val_transforms():
    transforms = [
        A.Resize(640, 640),
        ToTensorV2(p=1.0),
    ]
    return A.Compose(transforms, bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))