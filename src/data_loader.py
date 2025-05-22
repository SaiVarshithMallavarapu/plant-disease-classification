import tensorflow as tf
from tensorflow.keras import layers
preprocessing = layers

import os

from . import config 

def build_augmentation_pipeline():
    """Creates a sequential model for data augmentation."""
    if not config.APPLY_AUGMENTATION:
        return tf.keras.Sequential(name="augmentation_off")

    data_augmentation = tf.keras.Sequential(
        [
            preprocessing.RandomFlip("horizontal", input_shape=config.INPUT_SHAPE),
            preprocessing.RandomRotation(config.ROTATION_RANGE),
            preprocessing.RandomZoom(config.ZOOM_RANGE),
            preprocessing.RandomTranslation(height_factor=config.HEIGHT_SHIFT_RANGE, width_factor=config.WIDTH_SHIFT_RANGE),
            preprocessing.RandomContrast(factor=0.1), 
        ],
        name="data_augmentation",
    )
    print("Data augmentation pipeline built.")
    return data_augmentation


def get_preprocessing_function(model_choice):
    """Returns the appropriate preprocessing function for the chosen DenseNet model."""
    if model_choice == 'DenseNet121':
        return tf.keras.applications.densenet.preprocess_input
    elif model_choice == 'DenseNet201':
        return tf.keras.applications.densenet.preprocess_input
    else:
        raise ValueError(f"Preprocessing for {model_choice} not implemented.")

def load_datasets(model_choice):
    """
    Loads and preprocesses the training, validation, and test datasets.

    Returns:
        tuple: (train_dataset, validation_dataset, test_dataset, class_names)
    """
    print(f"Loading datasets from: {config.DATA_DIR}")
    print(f"Using model: {model_choice} for preprocessing")

    image_size = (config.IMG_HEIGHT, config.IMG_WIDTH)
    preprocess_input = get_preprocessing_function(model_choice)
    data_augmentation = build_augmentation_pipeline()

    # --- Load Training Data ---
    print(f"Loading training data from: {config.TRAIN_DIR}")
    train_dataset = tf.keras.utils.image_dataset_from_directory(
        config.TRAIN_DIR,
        shuffle=True,
        batch_size=config.BATCH_SIZE,
        image_size=image_size,
        label_mode='int'
    )
    config.CLASS_NAMES = train_dataset.class_names
    config.NUM_CLASSES = len(config.CLASS_NAMES)
    print(f"Found {config.NUM_CLASSES} classes: {config.CLASS_NAMES}")
    print(f"Number of training batches: {tf.data.experimental.cardinality(train_dataset)}")

    # --- Load Validation Data ---
    print(f"Loading validation data from: {config.VAL_DIR}")
    validation_dataset = tf.keras.utils.image_dataset_from_directory(
        config.VAL_DIR,
        shuffle=False, # No shuffling for validation
        batch_size=config.BATCH_SIZE,
        image_size=image_size,
        label_mode='int'
    )
    print(f"Number of validation batches: {tf.data.experimental.cardinality(validation_dataset)}")

    # --- Load Test Data ---
    print(f"Loading test data from: {config.TEST_DIR}")
    test_dataset = tf.keras.utils.image_dataset_from_directory(
        config.TEST_DIR,
        shuffle=False, # No shuffling for test
        batch_size=config.BATCH_SIZE,
        image_size=image_size,
        label_mode='int'
    )
    print(f"Number of test batches: {tf.data.experimental.cardinality(test_dataset)}")

    # --- Configure datasets for performance ---
    AUTOTUNE = tf.data.AUTOTUNE

    def prepare(ds, augment=False, shuffle_buffer=config.BUFFER_SIZE):
        if augment and config.APPLY_AUGMENTATION:
            ds = ds.map(lambda x, y: (data_augmentation(x, training=True), y),
                        num_parallel_calls=AUTOTUNE)

        ds = ds.map(lambda x, y: (preprocess_input(x), y),
                    num_parallel_calls=AUTOTUNE)

        ds = ds.cache().prefetch(buffer_size=AUTOTUNE)
        return ds

    # Prepare the datasets
    train_dataset = prepare(train_dataset, augment=True)
    validation_dataset = prepare(validation_dataset) # No augmentation for validation
    test_dataset = prepare(test_dataset)       # No augmentation for test

    print("Datasets loaded and preprocessing configured.")

    return train_dataset, validation_dataset, test_dataset, config.CLASS_NAMES

if __name__ == '__main__':
    print("Testing data loader...")
    if not hasattr(config, 'MODEL_CHOICE'):
         config.MODEL_CHOICE = 'DenseNet121'
    train_ds, val_ds, test_ds, class_names = load_datasets(config.MODEL_CHOICE)
    print(f"Successfully loaded datasets. Detected {len(class_names)} classes:")
    print(class_names)
    for images, labels in train_ds.take(1):
        print("Train batch shape:", images.shape)
        print("Train label shape:", labels.shape)
    for images, labels in val_ds.take(1):
        print("Validation batch shape:", images.shape)
        print("Validation label shape:", labels.shape)
    for images, labels in test_ds.take(1):
        print("Test batch shape:", images.shape)
        print("Test label shape:", labels.shape)