# src/config.py
import os

# Get the absolute path to the project root (where main.py is)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(BASE_DIR, 'data')
TRAIN_DIR = os.path.join(DATA_DIR, 'train')
VAL_DIR = os.path.join(DATA_DIR, 'validation')
TEST_DIR = os.path.join(DATA_DIR, 'test')
# --- Image Configuration ---
IMG_HEIGHT = 224  # DenseNet typical input size
IMG_WIDTH = 224
IMG_CHANNELS = 3
INPUT_SHAPE = (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)

# --- Training Configuration ---
BATCH_SIZE = 32  # Adjust based on your GPU memory
INITIAL_EPOCHS = 15 # Epochs for training the head only
FINE_TUNE_EPOCHS = 10 # Epochs for fine-tuning
TOTAL_EPOCHS = INITIAL_EPOCHS + FINE_TUNE_EPOCHS
BUFFER_SIZE = 1000 # For shuffling dataset

# Learning Rates
INITIAL_LEARNING_RATE = 1e-3 # Learning rate for the head training
FINE_TUNE_LEARNING_RATE = 1e-5 # Learning rate for fine-tuning (must be smaller)

# --- Model Configuration ---
# Options: 'DenseNet121', 'DenseNet201'
MODEL_CHOICE = 'DenseNet121' # Default, can be overridden

# Where to save models and results, specific to the chosen model
SAVED_MODELS_DIR = f'../saved_models/{MODEL_CHOICE}'
RESULTS_DIR = f'../results/{MODEL_CHOICE}'

# Create directories if they don't exist
os.makedirs(SAVED_MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# --- Fine-Tuning Configuration ---
# Number of layers to unfreeze from the end of the base model during fine-tuning
# For DenseNet121 (around 427 layers), unfreezing last ~40-50 might be a start
# For DenseNet201 (around 707 layers), unfreezing last ~60-70 might be a start
# Adjust this based on experimentation
FINE_TUNE_AT = 50 # Example: Unfreeze layers from this index onwards in the base model

# --- Augmentation Configuration ---
# Set to True to enable data augmentation for the training set
APPLY_AUGMENTATION = True
ROTATION_RANGE = 0.2
ZOOM_RANGE = 0.2
WIDTH_SHIFT_RANGE = 0.2
HEIGHT_SHIFT_RANGE = 0.2
SHEAR_RANGE = 0.2
HORIZONTAL_FLIP = True
FILL_MODE = 'nearest' # How to fill pixels created by transformations

# --- Calculated ---
# Will be determined dynamically in data_loader.py
NUM_CLASSES = None
CLASS_NAMES = None