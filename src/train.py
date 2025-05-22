# src/train.py
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
import os
import time

# Import project modules using relative paths
from . import config
from . import data_loader
from . import model_builder
from . import utils

def train_model(model_choice):
    """
    Loads data, builds, compiles, and trains the specified model.

    Args:
        model_choice (str): 'DenseNet121' or 'DenseNet201'.
    """
    print(f"--- Starting Training for: {model_choice} ---")

    # Set model-specific paths in config dynamically
    config.MODEL_CHOICE = model_choice
    config.SAVED_MODELS_DIR = f'../saved_models/{model_choice}'
    config.RESULTS_DIR = f'../results/{model_choice}'
    os.makedirs(config.SAVED_MODELS_DIR, exist_ok=True)
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    print(f"Model artifacts will be saved in: {config.SAVED_MODELS_DIR}")
    print(f"Results will be saved in: {config.RESULTS_DIR}")


    # --- 1. Load Data ---
    train_ds, val_ds, _, class_names = data_loader.load_datasets(model_choice)
    # Ensure NUM_CLASSES is set in config (should be done by load_datasets)
    if config.NUM_CLASSES is None:
         raise RuntimeError("Number of classes was not set by the data loader.")
    print(f"Data loaded. Number of classes: {config.NUM_CLASSES}")

    # --- 2. Build Model ---
    model = model_builder.build_model(model_choice)
    base_model = model.layers[1] # Assumes base_model is the second layer after Input

    # --- 3. Compile Model (Initial Phase - Head Training) ---
    # Compile with a standard learning rate for the new head layers
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=config.INITIAL_LEARNING_RATE),
                  loss='sparse_categorical_crossentropy', # Use sparse for integer labels
                  metrics=['accuracy'])
    print("Model compiled for initial head training.")
    model.summary() # Show summary after compilation

    # --- 4. Define Callbacks ---
    log_dir = os.path.join(config.RESULTS_DIR, "logs", "fit", time.strftime("%Y%m%d-%H%M%S"))
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

    # Save the best model based on validation accuracy
    checkpoint_path = os.path.join(config.SAVED_MODELS_DIR, f'{model_choice}_best.keras') # Use .keras format
    model_checkpoint_callback = ModelCheckpoint(
        filepath=checkpoint_path,
        save_weights_only=False, # Save entire model
        monitor='val_accuracy',
        mode='max',
        save_best_only=True, # Only save when val_accuracy improves
        verbose=1)

    # Stop training early if validation loss doesn't improve
    early_stopping_callback = EarlyStopping(
        monitor='val_loss',
        patience=10, # Number of epochs with no improvement after which training will be stopped
        verbose=1,
        restore_best_weights=True) # Restore model weights from the epoch with the best val_loss

    # Reduce learning rate when validation loss plateaus
    reduce_lr_callback = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2, # Factor by which the learning rate will be reduced. new_lr = lr * factor
        patience=5, # Number of epochs with no improvement after which learning rate will be reduced.
        verbose=1,
        min_lr=1e-6) # Lower bound on the learning rate

    callbacks_list = [model_checkpoint_callback, early_stopping_callback, reduce_lr_callback, tensorboard_callback]

    # --- 5. Initial Training (Train only the head) ---
    print(f"\n--- Starting Initial Training (Training Head Only) for {config.INITIAL_EPOCHS} epochs ---")
    history = model.fit(
        train_ds,
        epochs=config.INITIAL_EPOCHS,
        validation_data=val_ds,
        callbacks=callbacks_list
    )

    # --- 6. Fine-Tuning Phase ---
    print(f"\n--- Starting Fine-Tuning Phase ---")

    # Unfreeze the base model layers
    base_model.trainable = True

    # How many layers to unfreeze? Unfreeze all layers from FINE_TUNE_AT onwards.
    # Layers before FINE_TUNE_AT will remain frozen.
    print(f"Unfreezing base model layers from layer index {config.FINE_TUNE_AT} onwards.")
    for layer in base_model.layers[:config.FINE_TUNE_AT]:
        layer.trainable = False
    # for i, layer in enumerate(base_model.layers):
    #      print(i, layer.name, layer.trainable) # Optional: check which layers are trainable


    # Re-compile the model with a very low learning rate for fine-tuning
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=config.FINE_TUNE_LEARNING_RATE), # Crucial: Use a low LR
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    print("Model re-compiled for fine-tuning with lower learning rate.")
    model.summary()

    print(f"--- Continuing Training (Fine-Tuning) for {config.FINE_TUNE_EPOCHS} epochs ---")
    # Continue training on the entire model (fine-tuning)
    # Note: The 'epochs' argument in fit() is the total number of epochs for this *call*
    # We set initial_epoch to continue the epoch count from the previous phase
    history_fine_tune = model.fit(
        train_ds,
        epochs=config.TOTAL_EPOCHS, # Train up to the total desired epochs
        initial_epoch=history.epoch[-1] + 1, # Start counting from where the initial training left off
        validation_data=val_ds,
        callbacks=callbacks_list # Reuse the callbacks (EarlyStopping might trigger)
    )

    # --- 7. Combine History and Plot ---
    # Append fine-tuning history to the initial history
    for key in history.history.keys():
         history.history[key].extend(history_fine_tune.history[key])

    utils.plot_training_history(history, config.INITIAL_EPOCHS, model_choice)

    # --- 8. Save the Final Trained Model ---
    # The ModelCheckpoint callback already saved the *best* model.
    # We can also save the final model state after fine-tuning if desired.
    final_model_path = os.path.join(config.SAVED_MODELS_DIR, f'{model_choice}_final.keras')
    model.save(final_model_path)
    print(f"Final model saved to {final_model_path}")

    print(f"--- Training Complete for: {model_choice} ---")
    return model # Return the trained model

if __name__ == '__main__':
    # Example of running training directly for a specific model
    selected_model = 'DenseNet121' # Or 'DenseNet201'
    # You might want to use argparse here to select the model via command line
    print(f"Running training script directly for model: {selected_model}")
    try:
        trained_model = train_model(selected_model)
        print("Training finished successfully.")
        # Optional: Perform evaluation immediately after training
        # from . import evaluate
        # evaluate.evaluate_trained_model(selected_model)
    except Exception as e:
        print(f"An error occurred during training: {e}")
        import traceback
        traceback.print_exc()