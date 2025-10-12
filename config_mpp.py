import os

# === Paths ===
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT_DIR, "data")
TRAIN_DIR = os.path.join(DATA_DIR, "train")
VAL_DIR = os.path.join(DATA_DIR, "valid")
TEST_DIR = os.path.join(DATA_DIR, "test")
CHECKPOINTS_DIR = os.path.join(ROOT_DIR, "checkpoints_mpp")

# === Model Configuration ===
MODEL_NAME = "yolox_mpp"  # YOLOX-M++ model
NUM_CLASSES = 5  # Disease + severity classes
NUM_SEVERITY_LEVELS = 5  # Number of severity levels
PHI = "m"  # Model size: s, m, l, x

# === Disease and Severity Classes ===
DISEASE_CLASSES = ['curl_stage1', 'curl_stage2', 'healthy', 'leaf_enation', 'sooty']
SEVERITY_LEVELS = ['stage1', 'stage2', 'healthy', 'moderate', 'severe']

# === Training Hyperparameters ===
BATCH_SIZE = 2  # Reduced for M++ model (more complex architecture)
NUM_EPOCHS = 100  # Maximum epochs (early stopping will prevent overfitting)
LEARNING_RATE = 5e-4  # Reduced learning rate for M++ model
WEIGHT_DECAY = 1e-4  # Weight decay
ALPHA_IOU = 2.0  # α-IoU loss parameter

# === Early Stopping Parameters ===
EARLY_STOPPING_PATIENCE = 30  # Stop if no improvement for 30 epochs
MIN_DELTA = 0.001  # Minimum improvement threshold
WARMUP_EPOCHS = 10  # Learning rate warmup epochs

# === Model Architecture Parameters ===
# Backbone
CSP_X_DEPTH_MULTIPLIER = 0.67  # Depth multiplier for CSP-X blocks
CSP_X_WIDTH_MULTIPLIER = 0.75  # Width multiplier for CSP-X blocks
CONVNEXT_LAYER_SCALE = 1e-6  # ConvNeXt layer scale initialization

# Neck
BIFPN_NUM_LAYERS = 3  # Number of BiFPN layers
ASFF_ENABLED = True  # Enable Adaptive Spatial Feature Fusion
DUAL_SPP_ENABLED = True  # Enable Dual SPP

# Head
DYHEAD_ENABLED = True  # Enable Dynamic Head
DUAL_BRANCH_ENABLED = True  # Enable Dual-Branch Head
SEVERITY_BRANCH_WEIGHT = 0.5  # Weight for severity branch loss

# === Attention Mechanisms ===
CBAM_ENABLED = True  # Enable CBAM attention
CHANNEL_ATTENTION_RATIO = 16  # Channel attention reduction ratio
SPATIAL_ATTENTION_KERNEL = 7  # Spatial attention kernel size

# === Regularization ===
DROPBLOCK_ENABLED = True  # Enable DropBlock
DROPBLOCK_PROB = 0.1  # DropBlock probability
DROPBLOCK_SIZE = 7  # DropBlock size

# === Convolution Types ===
USE_GHOST_CONV = True  # Use Ghost Convolution
USE_DEPTHWISE_SEPARABLE = True  # Use Depthwise Separable Convolution
USE_GROUP_NORM = True  # Use Group Normalization
USE_EVO_NORM = False  # Use EvoNorm (alternative to GroupNorm)

# === Output Scales ===
NUM_OUTPUT_SCALES = 4  # P2, P3, P4, P5 (4 scales instead of 3)
P2_SCALE = 4.0  # P2 scale (smallest objects)
P3_SCALE = 8.0  # P3 scale
P4_SCALE = 16.0  # P4 scale
P5_SCALE = 32.0  # P5 scale (largest objects)

# === Loss Configuration ===
# Detection Loss
DETECTION_LOSS_WEIGHT = 1.0  # Weight for detection loss
ALPHA_IOU_WEIGHT = 1.0  # Weight for α-IoU loss
FOCAL_LOSS_WEIGHT = 1.0  # Weight for focal loss

# Severity Loss
SEVERITY_LOSS_WEIGHT = 0.5  # Weight for severity loss
SEVERITY_FOCAL_GAMMA = 2.0  # Gamma for severity focal loss
SEVERITY_WEIGHTS = [1.0, 1.0, 1.0, 2.0, 1.0]  # Weights for different severity levels

# === Augmentation Parameters ===
AUGMENTATIONS = {
    "flip_prob": 0.5,
    "rotate_prob": 0.5,
    "brightness_contrast_prob": 0.5,
    "brightness_limit": 0.25,
    "contrast_limit": 0.25,
    "blur_prob": 0.1,
    "noise_prob": 0.1,
}

# === Logging and Save Options ===
SAVE_PREDICTION_EVERY = 5  # Save prediction image every N epochs
LOG_INTERVAL = 10  # Log loss every N iterations
SAVE_INTERVAL = 10  # Save checkpoint every N epochs

# === Hardware Configuration ===
NUM_WORKERS = 0  # Number of data loading workers (0 for Windows)
PIN_MEMORY = True  # Pin memory for faster data loading
MIXED_PRECISION = True  # Use mixed precision training

# === Model Complexity Metrics ===
# Expected parameters: ~25M (vs ~9M for YOLOX-s)
# Expected FLOPs: ~60G (vs ~27G for YOLOX-s)
# Expected memory: ~8GB (vs ~4GB for YOLOX-s)

# === Performance Expectations ===
# Expected mAP improvement: +5-10% over baseline YOLOX
# Expected severity accuracy: +15-20% over baseline
# Expected inference time: ~50ms (vs ~30ms for YOLOX-s)
# Expected training time: ~2x longer than YOLOX-s
