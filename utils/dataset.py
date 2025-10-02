import os
import cv2
import numpy as np
import torch
import xml.etree.ElementTree as ET
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
import random

# -------------------------------------------------------------------
# 1. DEFINE THE AUGMENTATION PIPELINES USING ALBUMENTATIONS
# -------------------------------------------------------------------


def get_train_augs(input_size):
    """
    Augmentations for the training set.
    """
    return A.Compose(
        [
            # We resize first to a fixed size. Subsequent augmentations are applied to this canvas.
            A.Resize(height=input_size[0], width=input_size[1]),
            # Add strong geometric and color augmentations
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.1,
                scale_limit=0.1,
                rotate_limit=15,
                p=0.5,
                border_mode=cv2.BORDER_CONSTANT,
            ),
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
            format="albumentations",  # [x_min, y_min, x_max, y_max] in normalized coordinates [0, 1]
            label_fields=["class_labels"],
            min_visibility=0.1,  # A box is kept if at least 10% of it is visible after augmentation
        ),
    )


def get_val_augs(input_size):
    """
    Transformations for the validation set. Only resizing, normalization, and tensor conversion.
    """
    return A.Compose(
        [
            A.Resize(height=input_size[0], width=input_size[1]),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(
            format="albumentations", label_fields=["class_labels"]
        ),
    )


def get_mosaic_augs(input_size):
    """
    Minimal augmentations for mosaic/mixup cases (already at target size).
    """
    return A.Compose(
        [
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(
            format="albumentations", label_fields=["class_labels"]
        ),
    )


# -------------------------------------------------------------------
# 3. MOSAIC AUGMENTATION
# -------------------------------------------------------------------


def mosaic_augmentation(dataset_obj, index, input_size):
    """
    Mosaic augmentation: combines 4 images into one.

    Args:
        dataset_obj: The dataset object to sample images from
        index: The primary image index
        input_size: Tuple of (height, width)

    Returns:
        mosaic_image: Combined image
        mosaic_boxes: Combined bounding boxes (normalized [0,1])
        mosaic_labels: Combined class labels
    """
    h, w = input_size

    # Get 3 additional random indices
    indices = [index] + random.choices(range(len(dataset_obj)), k=3)

    # Initialize mosaic image
    mosaic_img = np.zeros((h, w, 3), dtype=np.uint8)

    # Center point for mosaic
    yc, xc = h // 2, w // 2

    all_boxes = []
    all_labels = []

    for i, idx in enumerate(indices):
        # Load image without augmentation
        img_name = dataset_obj.image_files[idx]
        img_path = os.path.join(dataset_obj.data_dir, img_name)
        xml_path = os.path.join(
            dataset_obj.data_dir, os.path.splitext(img_name)[0] + ".xml"
        )

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = img.shape[:2]

        # Parse annotations
        targets_list = dataset_obj._parse_xml(xml_path)

        # Place images in mosaic quadrants
        if i == 0:  # top left
            x1a, y1a, x2a, y2a = 0, 0, xc, yc
            x1b, y1b, x2b, y2b = w - xc, h - yc, w, h
        elif i == 1:  # top right
            x1a, y1a, x2a, y2a = xc, 0, w, yc
            x1b, y1b, x2b, y2b = 0, h - yc, w - xc, h
        elif i == 2:  # bottom left
            x1a, y1a, x2a, y2a = 0, yc, xc, h
            x1b, y1b, x2b, y2b = w - xc, 0, w, h - yc
        else:  # bottom right
            x1a, y1a, x2a, y2a = xc, yc, w, h
            x1b, y1b, x2b, y2b = 0, 0, w - xc, h - yc

        # Resize image to fit quadrant
        img_resized = cv2.resize(img[y1b:y2b, x1b:x2b], (x2a - x1a, y2a - y1a))
        mosaic_img[y1a:y2a, x1a:x2a] = img_resized

        # Adjust bounding boxes
        if targets_list:
            for target in targets_list:
                x1, y1, x2, y2, cls = target

                # Scale boxes to fit quadrant
                scale_x = (x2a - x1a) / (x2b - x1b)
                scale_y = (y2a - y1a) / (y2b - y1b)

                new_x1 = x1a + (x1 - x1b) * scale_x
                new_y1 = y1a + (y1 - y1b) * scale_y
                new_x2 = x1a + (x2 - x1b) * scale_x
                new_y2 = y1a + (y2 - y1b) * scale_y

                # Clip to mosaic image bounds
                new_x1 = max(0, min(new_x1, w - 1))
                new_y1 = max(0, min(new_y1, h - 1))
                new_x2 = max(new_x1 + 1, min(new_x2, w))
                new_y2 = max(new_y1 + 1, min(new_y2, h))

                # Only keep valid boxes with minimum size
                if new_x2 > new_x1 + 1 and new_y2 > new_y1 + 1:
                    # Normalize to [0, 1] range
                    norm_x1 = new_x1 / w
                    norm_y1 = new_y1 / h
                    norm_x2 = new_x2 / w
                    norm_y2 = new_y2 / h

                    # Apply strict clipping to prevent floating point precision errors
                    norm_x1 = max(0.0, min(norm_x1, 1.0))
                    norm_y1 = max(0.0, min(norm_y1, 1.0))
                    norm_x2 = max(0.0, min(norm_x2, 1.0))
                    norm_y2 = max(0.0, min(norm_y2, 1.0))

                    all_boxes.append([norm_x1, norm_y1, norm_x2, norm_y2])
                    all_labels.append(cls)

    # Ensure consistent array shapes
    if len(all_boxes) > 0:
        boxes_array = np.array(all_boxes, dtype=np.float32)
        labels_array = np.array(all_labels, dtype=np.float32)
    else:
        boxes_array = np.zeros((0, 4), dtype=np.float32)
        labels_array = np.zeros(0, dtype=np.float32)

    return mosaic_img, boxes_array, labels_array


# -------------------------------------------------------------------
# 4. MIXUP AUGMENTATION
# -------------------------------------------------------------------


def mixup_augmentation(img1, boxes1, labels1, img2, boxes2, labels2, alpha=0.5):
    """
    Mixup augmentation: blends two images together.

    Args:
        img1, img2: Images to mix
        boxes1, boxes2: Bounding boxes for each image
        labels1, labels2: Class labels for each image
        alpha: Mixing ratio (0.5 = equal mix)

    Returns:
        mixed_image: Blended image
        mixed_boxes: Combined bounding boxes
        mixed_labels: Combined class labels
    """
    # Random mixing ratio
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 0.5

    # Blend images
    mixed_img = (lam * img1 + (1 - lam) * img2).astype(np.uint8)

    # Combine boxes and labels - ensure proper shapes
    if len(boxes1) > 0 and len(boxes2) > 0:
        mixed_boxes = np.vstack([boxes1, boxes2])
        mixed_labels = np.concatenate([labels1, labels2])
    elif len(boxes1) > 0:
        mixed_boxes = boxes1
        mixed_labels = labels1
    elif len(boxes2) > 0:
        mixed_boxes = boxes2
        mixed_labels = labels2
    else:
        mixed_boxes = np.zeros((0, 4), dtype=np.float32)
        mixed_labels = np.zeros(0, dtype=np.float32)

    return mixed_img, mixed_boxes, mixed_labels


# -------------------------------------------------------------------
# 2. UPDATE THE DATASET CLASS TO USE ALBUMENTATIONS
# -------------------------------------------------------------------


class CottonDiseaseDataset(Dataset):
    def __init__(
        self,
        data_dir,
        augmentations,
        input_size=(640, 640),
        use_mosaic=True,
        use_mixup=True,
        mosaic_prob=0.5,
        mixup_prob=0.5,
    ):
        """
        Args:
            data_dir: Path to directory containing images and XMLs
            augmentations: An Albumentations pipeline.
            input_size: Model input size (height, width)
            use_mosaic: Whether to use Mosaic augmentation
            use_mixup: Whether to use Mixup augmentation
            mosaic_prob: Probability of applying Mosaic
            mixup_prob: Probability of applying Mixup
        """
        self.data_dir = data_dir
        self.augmentations = augmentations
        self.input_size = input_size
        self.use_mosaic = use_mosaic
        self.use_mixup = use_mixup
        self.mosaic_prob = mosaic_prob
        self.mixup_prob = mixup_prob

        self.image_files = []
        for f in sorted(os.listdir(data_dir)):
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                xml_file = os.path.splitext(f)[0] + ".xml"
                if os.path.exists(os.path.join(data_dir, xml_file)):
                    self.image_files.append(f)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Apply Mosaic augmentation with probability
        if self.use_mosaic and random.random() < self.mosaic_prob:
            image, bboxes, class_labels = mosaic_augmentation(
                self, idx, self.input_size
            )

            # Apply Mixup after Mosaic with probability
            if self.use_mixup and random.random() < self.mixup_prob:
                # Get another image for mixup
                idx2 = random.randint(0, len(self) - 1)
                img2, boxes2, labels2 = mosaic_augmentation(self, idx2, self.input_size)
                image, bboxes, class_labels = mixup_augmentation(
                    image, bboxes, class_labels, img2, boxes2, labels2
                )

            # Normalize bboxes to [0, 1] range and apply strict clipping
            h, w = self.input_size
            if len(bboxes) > 0:
                # Normalize to [0, 1] range
                bboxes[:, 0] /= w  # x1
                bboxes[:, 1] /= h  # y1
                bboxes[:, 2] /= w  # x2
                bboxes[:, 3] /= h  # y2

                # Apply strict clipping to prevent floating point precision errors
                bboxes = np.clip(bboxes, 0.0, 1.0)

                # Filter out invalid boxes (ensure minimum size in normalized space)
                min_size = 1.0 / max(w, h)  # At least 1 pixel in normalized coordinates
                valid_mask = (bboxes[:, 2] > bboxes[:, 0] + min_size) & (
                    bboxes[:, 3] > bboxes[:, 1] + min_size
                )
                bboxes = bboxes[valid_mask]
                class_labels = class_labels[valid_mask]

                # Ensure we still have matching lengths after filtering
                assert len(bboxes) == len(class_labels), (
                    f"Mismatch after filtering: {len(bboxes)} boxes vs {len(class_labels)} labels"
                )

            # Convert to list for Albumentations
            if len(bboxes) > 0:
                bboxes = bboxes.tolist()
                class_labels = class_labels.tolist()
            else:
                bboxes = []
                class_labels = []

            # Final validation before augmentation
            assert len(bboxes) == len(class_labels), (
                f"Length mismatch before augmentation: {len(bboxes)} boxes vs {len(class_labels)} labels"
            )

            # Use minimal augmentation pipeline for mosaic (no resize needed)
            mosaic_augs = get_mosaic_augs(self.input_size)
            augmented = mosaic_augs(
                image=image, bboxes=bboxes, class_labels=class_labels
            )
            image = augmented["image"]
            bboxes = augmented["bboxes"]
            class_labels = augmented["class_labels"]

            # Re-assemble the targets tensor
            if len(bboxes) > 0:
                targets = torch.cat(
                    (
                        torch.tensor(bboxes, dtype=torch.float32),
                        torch.tensor(class_labels, dtype=torch.float32).unsqueeze(1),
                    ),
                    dim=1,
                )
            else:
                targets = torch.zeros((0, 5), dtype=torch.float32)

            return image, targets

        # Normal loading without Mosaic
        img_name = self.image_files[idx]
        img_path = os.path.join(self.data_dir, img_name)
        xml_path = os.path.join(self.data_dir, os.path.splitext(img_name)[0] + ".xml")

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

            # Clip the bounding box coordinates to be within the image dimensions.
            targets_np[:, 0] = np.clip(targets_np[:, 0], 0, orig_w - 1)  # x1
            targets_np[:, 1] = np.clip(targets_np[:, 1], 0, orig_h - 1)  # y1
            targets_np[:, 2] = np.clip(targets_np[:, 2], 1, orig_w)  # x2
            targets_np[:, 3] = np.clip(targets_np[:, 3], 1, orig_h)  # y2

        # Separate bboxes and class labels for Albumentations
        bboxes = targets_np[:, :4]
        class_labels = targets_np[:, 4]

        # Normalize bboxes to [0, 1] range for Albumentations
        if len(bboxes) > 0:
            bboxes[:, 0] /= orig_w  # x1
            bboxes[:, 1] /= orig_h  # y1
            bboxes[:, 2] /= orig_w  # x2
            bboxes[:, 3] /= orig_h  # y2

            # Apply strict clipping to prevent floating point precision errors
            bboxes = np.clip(bboxes, 0.0, 1.0)

        # Apply Mixup augmentation with probability (without Mosaic)
        if self.use_mixup and random.random() < self.mixup_prob:
            idx2 = random.randint(0, len(self) - 1)
            img_name2 = self.image_files[idx2]
            img_path2 = os.path.join(self.data_dir, img_name2)
            xml_path2 = os.path.join(
                self.data_dir, os.path.splitext(img_name2)[0] + ".xml"
            )

            img2 = cv2.imread(img_path2)
            img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
            img2 = cv2.resize(img2, (orig_w, orig_h))

            targets_list2 = self._parse_xml(xml_path2)
            if targets_list2:
                targets_np2 = np.array(targets_list2, dtype=np.float32)
                # Clip coordinates for second image
                targets_np2[:, 0] = np.clip(targets_np2[:, 0], 0, orig_w - 1)  # x1
                targets_np2[:, 1] = np.clip(targets_np2[:, 1], 0, orig_h - 1)  # y1
                targets_np2[:, 2] = np.clip(targets_np2[:, 2], 1, orig_w)  # x2
                targets_np2[:, 3] = np.clip(targets_np2[:, 3], 1, orig_h)  # y2

                boxes2 = targets_np2[:, :4]
                labels2 = targets_np2[:, 4]

                # Normalize boxes2 to [0, 1] range
                boxes2[:, 0] /= orig_w  # x1
                boxes2[:, 1] /= orig_h  # y1
                boxes2[:, 2] /= orig_w  # x2
                boxes2[:, 3] /= orig_h  # y2
                boxes2 = np.clip(boxes2, 0.0, 1.0)
            else:
                boxes2 = np.array([])
                labels2 = np.array([])

            image, bboxes, class_labels = mixup_augmentation(
                image, bboxes, class_labels, img2, boxes2, labels2
            )

        # Apply augmentations
        if self.augmentations:
            # Convert to list format for Albumentations
            if len(bboxes) > 0:
                bbox_list = bboxes.tolist()
                label_list = class_labels.tolist()
            else:
                bbox_list = []
                label_list = []

            augmented = self.augmentations(
                image=image, bboxes=bbox_list, class_labels=label_list
            )
            image = augmented["image"]
            bboxes = (
                np.array(augmented["bboxes"])
                if augmented["bboxes"]
                else np.zeros((0, 4))
            )
            class_labels = (
                np.array(augmented["class_labels"])
                if augmented["class_labels"]
                else np.zeros(0)
            )

        # Re-assemble the targets tensor
        if len(bboxes) > 0:
            targets = torch.cat(
                (
                    torch.tensor(bboxes, dtype=torch.float32),
                    torch.tensor(class_labels, dtype=torch.float32).unsqueeze(1),
                ),
                dim=1,
            )
        else:
            targets = torch.zeros((0, 5), dtype=torch.float32)

        return image, targets

    def _parse_xml(self, xml_path):
        tree = ET.parse(xml_path)
        root = tree.getroot()
        targets = []
        for obj in root.findall("object"):
            class_name = obj.find("name").text.lower().strip()
            class_id = self._class_name_to_id(class_name)

            bbox = obj.find("bndbox")
            x1 = float(bbox.find("xmin").text)
            y1 = float(bbox.find("ymin").text)
            x2 = float(bbox.find("xmax").text)
            y2 = float(bbox.find("ymax").text)

            # Ensure valid bounding box
            if x2 > x1 and y2 > y1:
                targets.append([x1, y1, x2, y2, class_id])
        return targets

    def _class_name_to_id(self, class_name):
        class_map = {"healthy": 0, "diseased": 1}
        return class_map.get(class_name, 0)
