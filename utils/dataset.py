import os
import cv2
import numpy as np
import torch
import xml.etree.ElementTree as ET
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

# -------------------------------------------------------------------
# 1. DEFINE THE AUGMENTATION PIPELINES USING ALBUMENTATIONS
# -------------------------------------------------------------------

def get_train_augs(input_size):
    """
    Augmentations for the training set.
    """
    return A.Compose([
        # We resize first to a fixed size. Subsequent augmentations are applied to this canvas.
        A.Resize(height=input_size[0], width=input_size[1]),
        
        # Add strong geometric and color augmentations
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5, border_mode=cv2.BORDER_CONSTANT),
        A.RandomBrightnessContrast(p=0.3),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
        A.Blur(blur_limit=3, p=0.1),
        A.GaussNoise(p=0.1),
        
        # Crucial steps: Normalize and convert to a PyTorch tensor
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ], 
    # This part is key: it tells Albumentations how to handle bounding boxes.
    bbox_params=A.BboxParams(
        format='pascal_voc', # [x_min, y_min, x_max, y_max]
        label_fields=['class_labels'],
        min_visibility=0.1 # A box is kept if at least 10% of it is visible after augmentation
    ))

def get_val_augs(input_size):
    """
    Transformations for the validation set. Only resizing, normalization, and tensor conversion.
    """
    return A.Compose([
        A.Resize(height=input_size[0], width=input_size[1]),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ], 
    bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))


# -------------------------------------------------------------------
# 2. UPDATE THE DATASET CLASS TO USE ALBUMENTATIONS
# -------------------------------------------------------------------

class CottonDiseaseDataset(Dataset):
    def __init__(self, data_dir, augmentations, input_size=(640, 640)):
        """
        Args:
            data_dir: Path to directory containing images and XMLs
            augmentations: An Albumentations pipeline.
            input_size: Model input size (height, width)
        """
        self.data_dir = data_dir
        self.augmentations = augmentations
        self.input_size = input_size
        
        self.image_files = []
        for f in sorted(os.listdir(data_dir)):
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                xml_file = os.path.splitext(f)[0] + '.xml'
                if os.path.exists(os.path.join(data_dir, xml_file)):
                    self.image_files.append(f)
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.data_dir, img_name)
        xml_path = os.path.join(self.data_dir, os.path.splitext(img_name)[0] + '.xml')
        
        # Load image with OpenCV
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # --- GET THE ACTUAL IMAGE DIMENSIONS ---
        orig_h, orig_w = image.shape[:2]
        
        # Parse XML to get bounding boxes and class labels
        targets_list = self._parse_xml(xml_path)
        
        if not targets_list:
            targets_np = np.zeros((0, 5), dtype=np.float32)
        else:
            targets_np = np.array(targets_list, dtype=np.float32)
        
            # --------------------- THE FIX IS HERE ---------------------
            # Clip the bounding box coordinates to be within the image dimensions.
            # This handles annotation errors where boxes are slightly outside the image.
            targets_np[:, 0] = np.clip(targets_np[:, 0], 0, orig_w)  # x1
            targets_np[:, 1] = np.clip(targets_np[:, 1], 0, orig_h)  # y1
            targets_np[:, 2] = np.clip(targets_np[:, 2], 0, orig_w)  # x2
            targets_np[:, 3] = np.clip(targets_np[:, 3], 0, orig_h)  # y2
            # -----------------------------------------------------------

        # Separate bboxes and class labels for Albumentations
        bboxes = targets_np[:, :4]
        class_labels = targets_np[:, 4]
        
        # Apply augmentations
        if self.augmentations:
            augmented = self.augmentations(image=image, bboxes=bboxes, class_labels=class_labels)
            image = augmented['image']
            bboxes = augmented['bboxes']
            class_labels = augmented['class_labels']
            
        # Re-assemble the targets tensor
        if len(bboxes) > 0:
            targets = torch.cat(
                (torch.tensor(bboxes, dtype=torch.float32), 
                 torch.tensor(class_labels, dtype=torch.float32).unsqueeze(1)), 
                dim=1
            )
        else:
            targets = torch.zeros((0, 5), dtype=torch.float32)
            
        return image, targets

    
    def _parse_xml(self, xml_path):
        tree = ET.parse(xml_path)
        root = tree.getroot()
        targets = []
        for obj in root.findall('object'):
            class_name = obj.find('name').text.lower().strip()
            class_id = self._class_name_to_id(class_name)
            
            bbox = obj.find('bndbox')
            x1 = float(bbox.find('xmin').text)
            y1 = float(bbox.find('ymin').text)
            x2 = float(bbox.find('xmax').text)
            y2 = float(bbox.find('ymax').text)
            
            # Ensure valid bounding box
            if x2 > x1 and y2 > y1:
                targets.append([x1, y1, x2, y2, class_id])
        return targets
    
    def _class_name_to_id(self, class_name):
        class_map = {'healthy': 0, 'diseased': 1}
        return class_map.get(class_name, 0)