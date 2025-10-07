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
NUM_CLASSES = 5  # curl_stage1, curl_stage2, healthy, leaf_enation, sooty
MODEL_NAME = "improved_yolox_m"  # Options: improved_yolox_s, improved_yolox_m, improved_yolox_l, improved_yolox_x

# === Training Hyperparameters ===
BATCH_SIZE = 4  # Reduced for larger YOLOX-M model (uses ~3x more memory than YOLOX-S)
NUM_EPOCHS = 200  # Increased for better convergence
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0005
ALPHA_IOU = 2.0  # For alpha-IoU loss

# === Augmentations ===
AUGMENTATIONS = {
    "flip_prob": 0.5,
    "rotate_prob": 0.5,
    "brightness_contrast_prob": 0.5,
    "brightness_limit": 0.25,
    "contrast_limit": 0.25,
}

# === Logging and Save Options ===
SAVE_PREDICTION_EVERY = 2  # Save prediction image every N epochs
LOG_INTERVAL = 10  # Log loss every N iterations
