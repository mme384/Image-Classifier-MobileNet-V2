#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
    FILE NAME: train.py
    AUTHOR: Michalis Meyer
    DATE CREATED: 
    DATE LAST MODIFIED: 
    PYTHON VERSION: 
    SAMPLE COMMAND LINE: 
    SCRIPT PURPOSE: Train neural network.
"""
# Import modules
import warnings # Import module to deal with warnings
import numpy as np # Import module to use numpy
import matplotlib.pyplot as plt # Import module to use matplotlib
import tensorflow as tf # Import module to use tensorflow
import tensorflow_datasets as tfds # Import module to use tensorflow datasets
import os # Import module to deal with path names used in image data generator
from tensorflow.keras.preprocessing.image import ImageDataGenerator # Import module to create image data generator
import tensorflow_hub as hub # Import module to import model from tensorflow Hub
import logging # Import module to define logger
import json # Import module to do label mapping
import datetime # Date and time for time stamp in file name
import pytz # Adjust time zone for time stamp in file name
from PIL import Image # Process image for prediction


def set_up_workspace():
    """
    Run code to setup up the workspace
    """
    # Set defaults for the workspace
    warnings.filterwarnings('ignore')

    # Magic command for inline plotting
    # %matplotlib inline
    # %config InlineBackend.figure_format = 'retina'

    # Disable progress bar to keep shell nice and tidy
    tfds.disable_progress_bar()

    # Define logger
    logger = tf.get_logger()
    logger.setLevel(logging.ERROR)

    # Print some details
    print('\nDETAILS')
    print('Using:')
    print('\t\u2022 TensorFlow version:', tf.__version__)
    print('\t\u2022 tf.keras version:', tf.keras.__version__)
    print('\t\u2022 Running on GPU' if tf.test.is_gpu_available() else '\t\u2022 GPU device not found. Running on CPU')

    print('\nDone setting up workspace...')

def load_data_set():
    """
    Load the dataset with TensorFlow Datasets and create a training set, a validation set and a test set
    # https://www.tensorflow.org/datasets/catalog/overview
    # https://www.tensorflow.org/datasets/catalog/oxford_flowers102
    """
    # Define global variables
    global dataset, dataset_info, training_set, validation_set, test_set, class_names

    # Load the dataset with TensorFlow Datasets.
    dataset, dataset_info = tfds.load('oxford_flowers102', as_supervised=True, shuffle_files=True, with_info=True)

    # Create a training set, a validation set and a test set.
    training_set, validation_set, test_set = dataset['train'], dataset['validation'], dataset['test']

    # Load mapping from label to category name
    with open('label_map.json', 'r') as f:
        class_names = json.load(f)

    print('\nDone loading data...')

def explore_dataset():
    """
    Explore the loaded dada set
    """
    # Define global variables
    global num_classes

    # Display dataset_info
    dataset_info

    # Get the number of examples in each set from the dataset info.
    train_num_examples = dataset_info.splits['train'].num_examples
    validation_num_examples = dataset_info.splits['validation'].num_examples
    test_num_examples = dataset_info.splits['test'].num_examples
    # Get the number of classes in the dataset from the dataset info.
    num_classes = dataset_info.features['label'].num_classes

    # Print the above values
    print('There are {:,} images in the training set'.format(train_num_examples))
    print('There are {:,} images in the validation set'.format(validation_num_examples))
    print('There are {:,} images in the test set'.format(test_num_examples))
    print('There are {:,} classes in the data set'.format(num_classes))

    # Print the shape and corresponding label of 3 images in the training set.
    for image, label in training_set.take(3):
        image = image.numpy().squeeze()
        label = label.numpy()
        print('The shape of the image is', image.shape)
        print('The label of the image is', label)

    # Plot 1 image from the training set. Set the title
    # of the plot to the corresponding class name.
    for image, label in training_set.take(1):
        image = image.numpy().squeeze()
        label = label.numpy()
        plt.imshow(image)
        plt.title(class_names[str(label)])
        plt.show()

    print('\nDone exploring data set...')

def create_pipeline():
    """
    Create a pipeline for the training, validation and testing set
    """
    # Define global variables
    global batch_size, image_size, training_batches, validation_batches, testing_batches

    # Define batch size and image size
    batch_size = 64
    image_size = 224

    # Define function to convert images to appropriate format, resize to fit the input layer and normalize it
    def format_image(image, label):
        image = tf.cast(image, tf.float32)
        image = tf.image.resize(image, [image_size, image_size])
        image /= 255
        return image, label

    # Define batches, while modifying images according to above function as well as batch and prefetch them
    training_batches = training_set.map(format_image).batch(batch_size).prefetch(1)
    validation_batches = validation_set.map(format_image).batch(batch_size).prefetch(1)
    testing_batches = test_set.map(format_image).batch(batch_size).prefetch(1)

    print('\nDone creating pipeline...')

def train_classifier():
    """
    Build and train your network.
    """
    # Define global variables
    global model, history

    # URL to the MobileNet pre-trained model
    URL = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4"

    # Load pre-trained MobileNet feature extractor and wrap it into a Keras layer to be used as the first layer in the model
    feature_extractor = hub.KerasLayer(URL, input_shape=(image_size, image_size, 3))

    # Set the feature extractor as untrainable (as it is pre-trained)
    feature_extractor.trainable = False

    # Define dropout rate
    dropout_rate = 0.2

    # Number of epochs. The high number is on purpose, as early stopping is implemented.
    num_max_epochs = 3

    # Build model
    model = tf.keras.Sequential([feature_extractor,
                                 tf.keras.layers.Dense(512, activation='relu'),
                                 tf.keras.layers.Dropout(dropout_rate),
                                 tf.keras.layers.Dense(256, activation='relu'),
                                 tf.keras.layers.Dropout(dropout_rate),
                                 tf.keras.layers.Dense(num_classes, activation='softmax')])
    # Print model summary
    model.summary()

    # Set parameters useed to train model
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # Define callback function for early stopping
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                      patience=5)

    # Save best model
    # Path and filename with time stamp to save best model
    path_and_filename = './model_' + datetime.datetime.now(pytz.timezone('Europe/Zurich')).strftime(
        "%Y%m%d_%H%M%S") + '.h5'
    # Define callback function to save best model
    save_best_model = tf.keras.callbacks.ModelCheckpoint(path_and_filename,
                                                         monitor='val_loss',
                                                         save_best_only=True)

    history = model.fit(training_batches,
                        epochs=num_max_epochs,
                        validation_data=validation_batches,
                        callbacks=[early_stopping, save_best_model])

    print('\nDone training classifier...')

def main():
    """
    main function
    """
    # Set up the workspace
    set_up_workspace()

    # Load dataset and create training set, validation set and test set
    load_data_set()

    # Explore loaded data set
    explore_dataset()

    # Create a pipeline for the training, validation and testing set
    create_pipeline()

    # Build and train model
    train_classifier()

    # Indicate and of script
    print('End of script')

# Run main function
if __name__ == '__main__':
    main()