# src/model_builder.py
import tensorflow as tf
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.applications import DenseNet121, DenseNet201

from . import config # Import from the same directory

def get_base_model(model_choice):
    """Loads the pre-trained DenseNet base model."""
    if model_choice == 'DenseNet121':
        base_model = DenseNet121(input_shape=config.INPUT_SHAPE,
                                 include_top=False, # Exclude the final classification layer
                                 weights='imagenet')
        print("Loaded DenseNet121 base model with ImageNet weights.")
    elif model_choice == 'DenseNet201':
        base_model = DenseNet201(input_shape=config.INPUT_SHAPE,
                                 include_top=False,
                                 weights='imagenet')
        print("Loaded DenseNet201 base model with ImageNet weights.")
    else:
        raise ValueError(f"Invalid model_choice: {model_choice}. Choose 'DenseNet121' or 'DenseNet201'.")

    # Freeze the base model layers initially
    base_model.trainable = False
    print(f"Base model '{model_choice}' frozen.")
    return base_model

def build_model(model_choice):
    """
    Builds the full model by adding a custom head to the pre-trained base.

    Args:
        model_choice (str): 'DenseNet121' or 'DenseNet201'.

    Returns:
        tf.keras.Model: The compiled Keras model.
    """
    if config.NUM_CLASSES is None:
        raise ValueError("Number of classes (NUM_CLASSES) not set. Run data_loader first.")

    base_model = get_base_model(model_choice)

    # Define the input layer matching the base model's expectations
    inputs = tf.keras.Input(shape=config.INPUT_SHAPE)

    # Pass inputs through the base model
    x = base_model(inputs, training=False) # Use training=False for frozen base

    # --- Add Custom Layers ---
    # Global Average Pooling is standard practice after convolutional base
    x = GlobalAveragePooling2D(name='global_avg_pool')(x)

    # Add a Batch Normalization layer - can help stabilize training
    x = BatchNormalization(name='head_batch_norm_1')(x)

    # Add a dense layer with ReLU activation
    x = Dense(512, activation='relu', name='head_dense_1')(x) # Example size, tune if needed

    # Add Dropout for regularization to prevent overfitting
    x = Dropout(0.5, name='head_dropout')(x) # 0.5 is a common starting point

    # Add another dense layer (optional, can sometimes improve performance)
    # x = Dense(256, activation='relu', name='head_dense_2')(x)
    # x = Dropout(0.3, name='head_dropout_2')(x)

    # --- Output Layer ---
    # Final Dense layer with number of units equal to number of classes
    # Use 'softmax' activation for multi-class classification
    outputs = Dense(config.NUM_CLASSES, activation='softmax', name='output_predictions')(x)

    # --- Create the full model ---
    model = Model(inputs, outputs, name=f"{model_choice}_SoybeanClassifier")

    print(f"Model '{model.name}' built successfully.")

    return model

if __name__ == '__main__':
    print("Testing model builder...")
    if not hasattr(config, 'NUM_CLASSES') or config.NUM_CLASSES is None:
        config.NUM_CLASSES = 5 # Example number of classes
        print(f"Warning: NUM_CLASSES not set, using default: {config.NUM_CLASSES}")

    try:
        print("\nBuilding DenseNet121...")
        model121 = build_model('DenseNet121')

        print("\nBuilding DenseNet201...")
        model201 = build_model('DenseNet201')

        print("\nModel building test successful.")

    except Exception as e:
        print(f"An error occurred during model building test: {e}")