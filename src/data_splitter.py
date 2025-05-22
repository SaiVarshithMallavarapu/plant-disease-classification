import os
import shutil
from sklearn.model_selection import train_test_split

# Paths
FULL_DATA_DIR = '../data/full_data'
TRAIN_DIR = '../data/train'
VAL_DIR = '../data/validation'
TEST_DIR = '../data/test'

# Split ratios
TRAIN_RATIO = 0.7
VAL_RATIO = 0.2
TEST_RATIO = 0.1

def split_data():
    """
    Splits the dataset in FULL_DATA_DIR into train, validation, and test sets.
    """
    if not os.path.exists(FULL_DATA_DIR):
        raise FileNotFoundError(f"Full dataset directory not found: {FULL_DATA_DIR}")

    # Create train, validation, and test directories if they don't exist
    for dir_path in [TRAIN_DIR, VAL_DIR, TEST_DIR]:
        os.makedirs(dir_path, exist_ok=True)

    # Iterate through each class folder in the full dataset
    for class_name in os.listdir(FULL_DATA_DIR):
        class_dir = os.path.join(FULL_DATA_DIR, class_name)
        if not os.path.isdir(class_dir):
            continue

        # Get all file paths in the class directory
        file_paths = [os.path.join(class_dir, f) for f in os.listdir(class_dir) if os.path.isfile(os.path.join(class_dir, f))]

        # Split into train, validation, and test sets
        train_files, temp_files = train_test_split(file_paths, test_size=(1 - TRAIN_RATIO), random_state=42)
        val_files, test_files = train_test_split(temp_files, test_size=(TEST_RATIO / (VAL_RATIO + TEST_RATIO)), random_state=42)

        # Copy files to their respective directories
        for file_set, target_dir in zip([train_files, val_files, test_files], [TRAIN_DIR, VAL_DIR, TEST_DIR]):
            class_target_dir = os.path.join(target_dir, class_name)
            os.makedirs(class_target_dir, exist_ok=True)
            for file_path in file_set:
                shutil.copy(file_path, os.path.join(class_target_dir, os.path.basename(file_path)))

        print(f"Class '{class_name}' split into train, validation, and test sets.")

    print("Dataset splitting complete.")

if __name__ == '__main__':
    split_data()