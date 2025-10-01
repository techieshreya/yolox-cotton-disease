import os

# === Paths ===
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))  # Project root
DATA_DIR = os.path.join(ROOT_DIR, "data")
TRAIN_DIR = os.path.join(DATA_DIR, "train")
VAL_DIR = os.path.join(DATA_DIR, "valid")  # Corrected from 'val' to 'valid'
TEST_DIR = os.path.join(DATA_DIR, "test")
CHECKPOINTS_DIR = os.path.join(ROOT_DIR, "checkpoints")

# Annotation JSON paths (generated from XML using xml_to_coco.py)
TRAIN_ANN = os.path.join(DATA_DIR, "train.json")
VAL_ANN = os.path.join(DATA_DIR, "valid.json")
TEST_ANN = os.path.join(DATA_DIR, "test.json")

# === Model ===
NUM_CLASSES = 4  # e.g. ['Disease1', 'Disease2', 'Disease3', 'Healthy']
MODEL_NAME = "improved_yolox_s"  # Options: improved_yolox_s, improved_yolox_m, etc.

# === Training Hyperparameters ===
BATCH_SIZE = 32  # Adjust based on your GPU
NUM_EPOCHS = 100
LEARNING_RATE = 0.01
WEIGHT_DECAY = 0.0005
ALPHA_IOU = 3.0  # For alpha-CIoU loss (α=3)

# === Augmentations ===
AUGMENTATIONS = {
    "flip_prob": 0.5,
    "rotate_prob": 0.5,
    "brightness_contrast_prob": 0.5,
    "brightness_limit": 0.25,
    "contrast_limit": 0.25,
    "use_mosaic": True,  # Enable Mosaic augmentation
    "mosaic_prob": 0.5,  # Probability of applying Mosaic
    "use_mixup": True,  # Enable Mixup augmentation
    "mixup_prob": 0.5,  # Probability of applying Mixup
}

# === Logging and Save Options ===
SAVE_PREDICTION_EVERY = 2  # Save prediction image every N epochs
LOG_INTERVAL = 10  # Log loss every N iterations
