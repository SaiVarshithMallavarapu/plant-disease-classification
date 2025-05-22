# src/evaluate.py
import tensorflow as tf
import os
import numpy as np

from . import config
from . import data_loader
from . import utils

def evaluate_trained_model(model_choice):
    """
    Loads the best saved model and evaluates it on the test set.

    Args:
        model_choice (str): 'DenseNet121' or 'DenseNet201'.
    """
    print(f"--- Evaluating Model: {model_choice} ---")

    config.MODEL_CHOICE = model_choice
    config.SAVED_MODELS_DIR = f'../saved_models/{model_choice}'
    config.RESULTS_DIR = f'../results/{model_choice}'
    os.makedirs(config.RESULTS_DIR, exist_ok=True)

    # --- 1. Load Test Data ---
    _, _, test_ds, class_names = data_loader.load_datasets(model_choice)
    if class_names is None:
        raise RuntimeError("Class names not loaded. Ensure data loader runs first or is called.")
    print(f"Test data loaded. Found {len(class_names)} classes.")


    # --- 2. Load the Saved Model ---
    model_path = os.path.join(config.SAVED_MODELS_DIR, f'{model_choice}_best.keras')
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        print("Please ensure the model has been trained and saved correctly.")
        model_path = os.path.join(config.SAVED_MODELS_DIR, f'{model_choice}_final.keras')
        if not os.path.exists(model_path):
             print(f"Error: Final model file also not found at {model_path}")
             return
        else:
            print(f"Warning: Loading final model state from {model_path} instead of the best one.")

    print(f"Loading trained model from: {model_path}")
    try:
        model = tf.keras.models.load_model(model_path)
        print("Model loaded successfully.")
        model.summary() 
    except Exception as e:
        print(f"Error loading model: {e}")
        print("If you used custom layers/objects, you might need to provide the 'custom_objects' argument to load_model.")
        import traceback
        traceback.print_exc()
        return

    # --- 3. Evaluate the Model ---
    utils.evaluate_model(model, test_ds, class_names, model_choice)

if __name__ == '__main__':
    selected_model = 'DenseNet121' 
    print(f"Running evaluation script directly for model: {selected_model}")
    try:
        evaluate_trained_model(selected_model)
        print("Evaluation finished successfully.")
    except Exception as e:
        print(f"An error occurred during evaluation: {e}")
        import traceback
        traceback.print_exc()