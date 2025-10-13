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
    def __init__(self, data_dir, augmentations, input_size=(640, 640), mosaic_prob=0.5, mixup_prob=0.15, enable_mosaic=True, enable_mixup=True):
        """
        Args:
            data_dir: Path to directory containing images and XMLs
            augmentations: An Albumentations pipeline.
            input_size: Model input size (height, width)
        """
        self.data_dir = data_dir
        self.augmentations = augmentations
        self.input_size = input_size
        self.mosaic_prob = mosaic_prob
        self.mixup_prob = mixup_prob
        self.enable_mosaic = enable_mosaic
        self.enable_mixup = enable_mixup
        
        self.image_files = []
        for f in sorted(os.listdir(data_dir)):
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                xml_file = os.path.splitext(f)[0] + '.xml'
                if os.path.exists(os.path.join(data_dir, xml_file)):
                    self.image_files.append(f)

        # Precompute class frequencies and per-sample sampling weights for class-balanced sampling
        self.class_freq = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
        self.sample_weights = []
        for img_name in self.image_files:
            xml_path = os.path.join(self.data_dir, os.path.splitext(img_name)[0] + '.xml')
            targets_list = self._parse_xml(xml_path)
            if targets_list:
                labels = [int(t[4]) for t in targets_list]
                for c in set(labels):
                    self.class_freq[c] += labels.count(c)
        # avoid zero
        for k in self.class_freq:
            if self.class_freq[k] == 0:
                self.class_freq[k] = 1
        for img_name in self.image_files:
            xml_path = os.path.join(self.data_dir, os.path.splitext(img_name)[0] + '.xml')
            targets_list = self._parse_xml(xml_path)
            if not targets_list:
                self.sample_weights.append(1.0)
            else:
                labels = [int(t[4]) for t in targets_list]
                # weight is mean of inverse frequency
                invs = [1.0 / float(self.class_freq[c]) for c in labels]
                self.sample_weights.append(float(sum(invs) / len(invs)))
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # Decide augmentation strategy
        use_mosaic = self.enable_mosaic and (np.random.rand() < self.mosaic_prob)
        if use_mosaic and len(self.image_files) >= 4:
            image, targets_np = self._load_mosaic_image_and_targets(idx)
            # Optional MixUp after Mosaic
            if self.enable_mixup and (np.random.rand() < self.mixup_prob) and len(self.image_files) >= 1:
                mix_idx = np.random.randint(0, len(self.image_files))
                mix_image, mix_targets = self._load_image_and_targets(mix_idx)
                image, targets_np = self._mixup(image, targets_np, mix_image, mix_targets)
        else:
            image, targets_np = self._load_image_and_targets(idx)

        # Apply albumentations at the end (resize/normalize/tensor + light augs)
        bboxes = targets_np[:, :4] if targets_np.size > 0 else []
        class_labels = targets_np[:, 4] if targets_np.size > 0 else []
        
        # Validate bounding boxes before augmentation
        if len(bboxes) > 0:
            valid_bboxes = []
            valid_labels = []
            for i, bbox in enumerate(bboxes):
                x_min, y_min, x_max, y_max = bbox
                # Check if bbox is valid (x_max > x_min and y_max > y_min)
                if x_max > x_min and y_max > y_min:
                    valid_bboxes.append(bbox)
                    valid_labels.append(class_labels[i])
            bboxes = valid_bboxes
            class_labels = valid_labels
        
        if self.augmentations:
            augmented = self.augmentations(image=image, bboxes=bboxes, class_labels=class_labels)
            image = augmented['image']
            bboxes = augmented['bboxes']
            class_labels = augmented['class_labels']

        if len(bboxes) > 0:
            targets = torch.cat(
                (torch.tensor(bboxes, dtype=torch.float32),
                 torch.tensor(class_labels, dtype=torch.float32).unsqueeze(1)),
                dim=1
            )
        else:
            targets = torch.zeros((0, 5), dtype=torch.float32)

        return image, targets

    def _load_image_and_targets(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.data_dir, img_name)
        xml_path = os.path.join(self.data_dir, os.path.splitext(img_name)[0] + '.xml')

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]
        targets_list = self._parse_xml(xml_path)
        if not targets_list:
            targets_np = np.zeros((0, 5), dtype=np.float32)
        else:
            targets_np = np.array(targets_list, dtype=np.float32)
            targets_np[:, 0] = np.clip(targets_np[:, 0], 0, w)
            targets_np[:, 1] = np.clip(targets_np[:, 1], 0, h)
            targets_np[:, 2] = np.clip(targets_np[:, 2], 0, w)
            targets_np[:, 3] = np.clip(targets_np[:, 3], 0, h)
            
            # Remove invalid bounding boxes after clipping
            bw = targets_np[:, 2] - targets_np[:, 0]
            bh = targets_np[:, 3] - targets_np[:, 1]
            keep = (bw > 2) & (bh > 2)
            targets_np = targets_np[keep]
        return image, targets_np

    def _load_mosaic_image_and_targets(self, idx):
        input_h, input_w = self.input_size
        yc, xc = [int(np.random.uniform(0.5 * s, 1.5 * s)) for s in (input_h, input_w)]

        indices = [idx] + [np.random.randint(0, len(self.image_files)) for _ in range(3)]
        mosaic_img = np.full((input_h * 2, input_w * 2, 3), 114, dtype=np.uint8)
        mosaic_targets = []

        for i, index in enumerate(indices):
            img, targets = self._load_image_and_targets(index)
            h, w = img.shape[:2]

            scale = np.random.uniform(0.4, 1.0)
            new_h, new_w = int(h * scale), int(w * scale)
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

            if i == 0:  # top left
                x1a, y1a, x2a, y2a = max(xc - new_w, 0), max(yc - new_h, 0), xc, yc
                x1b, y1b, x2b, y2b = new_w - (x2a - x1a), new_h - (y2a - y1a), new_w, new_h
            elif i == 1:  # top right
                x1a, y1a, x2a, y2a = xc, max(yc - new_h, 0), min(xc + new_w, 2 * input_w), yc
                x1b, y1b, x2b, y2b = 0, new_h - (y2a - y1a), min(new_w, x2a - x1a), new_h
            elif i == 2:  # bottom left
                x1a, y1a, x2a, y2a = max(xc - new_w, 0), yc, xc, min(yc + new_h, 2 * input_h)
                x1b, y1b, x2b, y2b = new_w - (x2a - x1a), 0, new_w, min(y2a - y1a, new_h)
            else:  # bottom right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + new_w, 2 * input_w), min(yc + new_h, 2 * input_h)
                x1b, y1b, x2b, y2b = 0, 0, min(new_w, x2a - x1a), min(new_h, y2a - y1a)

            mosaic_img[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]

            padw, padh = x1a - x1b, y1a - y1b
            if targets.size > 0:
                boxes = targets[:, :4].copy()
                classes = targets[:, 4:5].copy()
                boxes[:, 0] = boxes[:, 0] * (new_w / w) + padw
                boxes[:, 1] = boxes[:, 1] * (new_h / h) + padh
                boxes[:, 2] = boxes[:, 2] * (new_w / w) + padw
                boxes[:, 3] = boxes[:, 3] * (new_h / h) + padh
                merged = np.hstack((boxes, classes))
                mosaic_targets.append(merged)

        if len(mosaic_targets) > 0:
            mosaic_targets = np.concatenate(mosaic_targets, axis=0)
            # Clip boxes to the mosaic image
            np.clip(mosaic_targets[:, 0], 0, 2 * input_w, out=mosaic_targets[:, 0])
            np.clip(mosaic_targets[:, 1], 0, 2 * input_h, out=mosaic_targets[:, 1])
            np.clip(mosaic_targets[:, 2], 0, 2 * input_w, out=mosaic_targets[:, 2])
            np.clip(mosaic_targets[:, 3], 0, 2 * input_h, out=mosaic_targets[:, 3])
            # Remove invalid or tiny boxes
            bw = mosaic_targets[:, 2] - mosaic_targets[:, 0]
            bh = mosaic_targets[:, 3] - mosaic_targets[:, 1]
            keep = (bw > 2) & (bh > 2)
            mosaic_targets = mosaic_targets[keep]
        else:
            mosaic_targets = np.zeros((0, 5), dtype=np.float32)

        # Center crop to original input size
        x_start = max(xc - input_w // 2, 0)
        y_start = max(yc - input_h // 2, 0)
        x_end = x_start + input_w
        y_end = y_start + input_h
        mosaic_img = mosaic_img[y_start:y_end, x_start:x_end]

        if mosaic_targets.size > 0:
            mosaic_targets[:, [0, 2]] -= x_start
            mosaic_targets[:, [1, 3]] -= y_start
            mosaic_targets[:, 0] = np.clip(mosaic_targets[:, 0], 0, input_w)
            mosaic_targets[:, 1] = np.clip(mosaic_targets[:, 1], 0, input_h)
            mosaic_targets[:, 2] = np.clip(mosaic_targets[:, 2], 0, input_w)
            mosaic_targets[:, 3] = np.clip(mosaic_targets[:, 3], 0, input_h)
            
            # Remove invalid bounding boxes after clipping
            bw = mosaic_targets[:, 2] - mosaic_targets[:, 0]
            bh = mosaic_targets[:, 3] - mosaic_targets[:, 1]
            keep = (bw > 2) & (bh > 2)
            mosaic_targets = mosaic_targets[keep]

        return mosaic_img, mosaic_targets.astype(np.float32)

    def _mixup(self, img1, targets1, img2, targets2, alpha=0.2):
        lam = np.random.beta(alpha, alpha)
        h = max(img1.shape[0], img2.shape[0])
        w = max(img1.shape[1], img2.shape[1])
        out = np.zeros((h, w, 3), dtype=np.uint8)
        out[:img1.shape[0], :img1.shape[1]] = img1
        out[:img2.shape[0], :img2.shape[1]] = (lam * out[:img2.shape[0], :img2.shape[1]] + (1 - lam) * img2).astype(np.uint8)

        # Simply concatenate boxes; labels unaffected
        if targets1.size == 0 and targets2.size == 0:
            return out, np.zeros((0, 5), dtype=np.float32)
        if targets1.size == 0:
            return out, targets2
        if targets2.size == 0:
            return out, targets1
        merged = np.concatenate([targets1, targets2], axis=0)
        return out, merged.astype(np.float32)

    
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
        """
        Map class names to IDs for disease + severity detection.
        Based on the paper: "Handling Severity Levels of Multiple Co-Occurring Cotton Plant Diseases Using Improved YOLOX Model"
        """
        class_map = {
            # Disease types with severity levels
            'curl_stage1': 0,    # Cotton leaf curl disease - Stage 1 (mild)
            'curl_stage2': 1,    # Cotton leaf curl disease - Stage 2 (severe)
            'healthy': 2,        # Healthy cotton plant
            'leaf_enation': 3,   # Leaf enation disease
            'sooty': 4          # Sooty mold disease
        }
        return class_map.get(class_name, 2)  # Default to healthy if unknown
    
    def get_class_info(self, class_id):
        """
        Get disease and severity information for a class ID.
        Returns: (disease_type, severity_level)
        """
        class_info = {
            0: ('curl', 'stage1'),
            1: ('curl', 'stage2'), 
            2: ('healthy', 'none'),
            3: ('leaf_enation', 'moderate'),
            4: ('sooty', 'severe')
        }
        return class_info.get(class_id, ('unknown', 'unknown'))