import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.preprocessing import image
# from src import config  # Import project configuration
import sys
sys.path.append(r"C:\Users\mvars\Downloads\Minor Project")
from soybean_disease_classifier.src import config

CLASS_NAMES = [
    "Mossaic Virus",
    "Southern blight",
    "Sudden Death Syndrone",
    "Yellow Mosaic",
    "bacterial_blight",
    "brown_spot",
    "crestamento",
    "ferrugen",
    "powdery_mildew",
    "septoria"
]




def predict_image(model_choice, image_path):
    """
    Loads a saved model and predicts the class of a given image.

    Args:
        model_choice (str): 'DenseNet121' or 'DenseNet201'.
        image_path (str): Path to the image to predict.

    Returns:
        str: Predicted class name.
    """
    # Set model-specific paths
    # model_path = C:\Users\mvars\Downloads\Minor Project\saved_models\DenseNet121\DenseNet121_best.keras
    model_path = r"C:\Users\mvars\Downloads\Minor Project\saved_models\DenseNet121\DenseNet121_best.keras"
    # model_path = os.path.join(config.SAVED_MODELS_DIR, f"{model_choice}_best.keras")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")

    # Load the saved model
    print(f"Loading model from: {model_path}")
    model = tf.keras.models.load_model(model_path)
    print("Model loaded successfully.")

    # Load and preprocess the image
    img = image.load_img(image_path, target_size=(config.IMG_HEIGHT, config.IMG_WIDTH))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    preprocess_input = tf.keras.applications.densenet.preprocess_input
    img_array = preprocess_input(img_array)

    # Predict the class
    # predictions = model.predict(img_array)
    # predicted_class_index = np.argmax(predictions, axis=1)[0]
    # print(f"Predicted class index: {predicted_class_index}")
    # # predicted_class_name = config.CLASS_NAMES[predicted_class_index]
    # predicted_class_name = CLASS_NAMES[predicted_class_index]
    predictions = model.predict(img_array)[0]  # shape: (num_classes,)
    
    # Print all class probabilities
    print("\nClass probabilities:")
    for i, prob in enumerate(predictions):
        print(f"{CLASS_NAMES[i]:<25}: {prob * 100:.2f}%")

    predicted_class_index = np.argmax(predictions)
    predicted_class_name = CLASS_NAMES[predicted_class_index]
    confidence = predictions[predicted_class_index] * 100
    print(f"\nPredicted class: {predicted_class_name} ({confidence:.2f}% confidence)")




    # print(f"Predicted class: {predicted_class_name}")
    # confidence = predictions[0][predicted_class_index] * 100
    # print(f"Predicted class: {predicted_class_name} ({confidence:.2f}% confidence)")

    return predicted_class_name


if __name__ == "__main__":
    # Example usage
    model_choice = "DenseNet121"  # Change to 'DenseNet201' if needed
    # image_path = "C:\\Users\\mvars\\Downloads\\Minor Project\\soybean_disease_classifier\\data\\train\\brown_spot\\bs53.bmp"
    # image_path = "C:\\Users\\mvars\\Downloads\\Minor Project\\soybean_disease_classifier\\data\\train\\bacterial_blight\\bb.jpg"
    image_path = "C:\\Users\\mvars\\Downloads\\bbg.jpeg"
    

    # C:\Users\mvars\Downloads\Minor Project\soybean_disease_classifier\data\train\brown_spot\bs.jpg

    config.MODEL_CHOICE = model_choice

    # Ensure class names are loaded
    # if not config.CLASS_NAMES:
    #     raise RuntimeError("Class names not loaded. Ensure the data loader has been run.")

    predict_image(model_choice, image_path)