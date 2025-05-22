# Soybean Disease Classifier

This project aims to classify soybean diseases using deep learning techniques. The dataset consists of images of soybean plants affected by various diseases, and the model will be trained to identify these diseases based on the images provided.

## Project Structure

- **data/**: Contains the dataset divided into training, validation, and test sets.
  - **train/**: Training dataset images.
    - **disease_1/**: Images of disease 1.
    - **disease_2/**: Images of disease 2.
  - **validation/**: Validation dataset images.
    - **disease_1/**: Images of disease 1.
    - **disease_2/**: Images of disease 2.
  - **test/**: Test dataset images.
    - **disease_1/**: Images of disease 1.
    - **disease_2/**: Images of disease 2.

- **src/**: Source code for the project.
  - **config.py**: Configuration variables for the project.
  - **data_loader.py**: Script for loading and preprocessing the dataset.
  - **model_builder.py**: Script for defining the model architecture.
  - **train.py**: Script for training the model.
  - **evaluate.py**: Script for evaluating the model's performance.
  - **utils.py**: Utility functions for various tasks.

- **saved_models/**: Directory for saving trained models.
  - **densenet121/**: Saved model for DenseNet121.
  - **densenet201/**: Saved model for DenseNet201.

- **results/**: Directory for storing evaluation results.
  - **densenet121/**: Results related to DenseNet121.
  - **densenet201/**: Results related to DenseNet201.

- **requirements.txt**: Lists the dependencies required for the project.

- **main.py**: Main script to run the training and evaluation processes.

## Installation

To install the required dependencies, run:

```
pip install -r requirements.txt
```

## Usage


To train the model, execute one of the following commands:

```
python main.py train --model DenseNet121
python main.py train --model DenseNet201

```
To evaluate the model, execute one of the following commands:
```

python main.py evaluate --model DenseNet121
python main.py evaluate --model DenseNet201

```


