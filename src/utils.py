# src/utils.py
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import os

from . import config # Import from the same directory

def plot_training_history(history, initial_epochs, model_choice):
    """
    Plots the training and validation accuracy and loss.

    Args:
        history (tf.keras.callbacks.History): History object from model.fit().
        initial_epochs (int): Number of epochs trained before potential fine-tuning.
        model_choice (str): Name of the model for titles/saving.
    """
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    total_epochs = len(acc)
    epochs_range = range(total_epochs)

    plt.figure(figsize=(12, 6))

    # Plot Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    # Mark where fine-tuning started, if applicable
    if total_epochs > initial_epochs:
        plt.axvline(initial_epochs -1, linestyle='--', color='gray', label='Start Fine-Tuning') # -1 because epochs are 0-indexed in plot range
    plt.legend(loc='lower right')
    plt.title(f'{model_choice} - Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([min(plt.ylim())*0.9, 1.0]) # Adjust y-axis limits


    # Plot Loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    # Mark where fine-tuning started, if applicable
    if total_epochs > initial_epochs:
         plt.axvline(initial_epochs -1, linestyle='--', color='gray', label='Start Fine-Tuning')
    plt.legend(loc='upper right')
    plt.title(f'{model_choice} - Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.ylim([0, max(plt.ylim())*1.1]) # Adjust y-axis limits

    plt.suptitle(f'{model_choice} Training History', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title overlap

    # Save the plot
    plot_filename = os.path.join(config.RESULTS_DIR, f'{model_choice}_training_history.png')
    plt.savefig(plot_filename)
    print(f"Training history plot saved to {plot_filename}")
    # plt.show() # Optionally display the plot

def plot_confusion_matrix(y_true, y_pred, class_names, model_choice):
    """
    Plots a confusion matrix using Seaborn.

    Args:
        y_true (np.array): True labels.
        y_pred (np.array): Predicted labels (integers).
        class_names (list): List of class names for labels.
        model_choice (str): Name of the model for titles/saving.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f'{model_choice} - Confusion Matrix')
    plt.ylabel('Actual Class')
    plt.xlabel('Predicted Class')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()

    # Save the plot
    cm_filename = os.path.join(config.RESULTS_DIR, f'{model_choice}_confusion_matrix.png')
    plt.savefig(cm_filename)
    print(f"Confusion matrix plot saved to {cm_filename}")
    # plt.show() # Optionally display the plot

def evaluate_model(model, test_dataset, class_names, model_choice):
    """
    Evaluates the model on the test dataset and prints/saves results.

    Args:
        model (tf.keras.Model): The trained Keras model.
        test_dataset (tf.data.Dataset): The preprocessed test dataset.
        class_names (list): List of class names.
        model_choice (str): Name of the model for saving results.
    """
    print(f"\n--- Evaluating Model: {model_choice} ---")

    # 1. Evaluate using model.evaluate() - Gets loss and accuracy
    loss, accuracy = model.evaluate(test_dataset, verbose=1)
    print(f"\nTest Loss: {loss:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")

    # 2. Get predictions for detailed metrics
    print("Generating predictions on the test set...")
    # Iterate over the dataset to get all predictions and labels
    y_pred_probs = model.predict(test_dataset, verbose=1)
    y_pred = np.argmax(y_pred_probs, axis=1) # Get the class index with highest probability

    # Extract true labels from the test dataset
    y_true = []
    for images, labels in test_dataset.unbatch().batch(config.BATCH_SIZE): # Efficiently get all labels
        y_true.extend(labels.numpy())
    y_true = np.array(y_true)

    print(f"Number of test samples: {len(y_true)}")
    print(f"Number of predictions: {len(y_pred)}")

    if len(y_true) != len(y_pred):
        print("Warning: Mismatch between number of true labels and predictions!")
        # This might happen if the dataset size isn't perfectly divisible by batch size
        # and the dataset wasn't configured to drop the remainder.
        # For simplicity here, we'll proceed, but investigate if this occurs.

    # 3. Classification Report
    print("\nClassification Report:")
    report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    print(report)

    # Save classification report to file
    report_filename = os.path.join(config.RESULTS_DIR, f'{model_choice}_classification_report.txt')
    with open(report_filename, 'w') as f:
        f.write(f"Model: {model_choice}\n")
        f.write(f"Test Loss: {loss:.4f}\n")
        f.write(f"Test Accuracy: {accuracy:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report)
    print(f"Classification report saved to {report_filename}")

    # 4. Confusion Matrix Plot
    print("Plotting confusion matrix...")
    plot_confusion_matrix(y_true, y_pred, class_names, model_choice)

    print(f"--- Evaluation Complete for {model_choice} ---")
    return loss, accuracy