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
NUM_CLASSES = 5  # curl_stage1, curl_stage2, healthy, leaf_enation, sooty (includes disease + severity)
MODEL_NAME = "improved_yolox_s"  # Using YOLOX-s as base model with improvements
DISEASE_CLASSES = ['curl_stage1', 'curl_stage2', 'healthy', 'leaf_enation', 'sooty']
SEVERITY_LEVELS = ['stage1', 'stage2', 'healthy', 'moderate', 'severe']  # Severity mapping

# === Training Hyperparameters ===
BATCH_SIZE = 4  # Optimized for YOLOX-s with improved architecture
NUM_EPOCHS = 200  # Maximum epochs (early stopping will prevent overfitting)
LEARNING_RATE = 0.001  # Tuned learning rate for cotton disease detection
WEIGHT_DECAY = 0.0001  # Reduced weight decay for better convergence
ALPHA_IOU = 2.0  # α-IoU loss parameter (paper recommendation)

# === Early Stopping Parameters ===
EARLY_STOPPING_PATIENCE = 30  # Stop if no improvement for 30 epochs
MIN_DELTA = 0.001  # Minimum improvement threshold
WARMUP_EPOCHS = 10  # Learning rate warmup epochs

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
