# utils/transforms.py
import albumentations as A
from albumentations.pytorch import ToTensorV2
import config  # Import your config.py

def get_train_transforms():
    transforms = [
        A.RandomResizedCrop(640, 640, scale=(0.8, 1.2), ratio=(0.9, 1.1), p=1.0),
        A.HorizontalFlip(p=config.AUGMENTATIONS['flip_prob']),
        A.RandomRotate90(p=config.AUGMENTATIONS['rotate_prob']),
        A.RandomBrightnessContrast(
            p=config.AUGMENTATIONS['brightness_contrast_prob'],
            brightness_limit=config.AUGMENTATIONS['brightness_limit'],
            contrast_limit=config.AUGMENTATIONS['contrast_limit'],
        ),
        A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=10, val_shift_limit=10, p=0.3),
        ToTensorV2(p=1.0),  # Convert to PyTorch tensor
    ]
    return A.Compose(transforms, bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))


def get_val_transforms():
    transforms = [
        A.Resize(640, 640),
        ToTensorV2(p=1.0),
    ]
    return A.Compose(transforms, bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))