#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
    FILE NAME: train.py
    AUTHOR: Michalis Meyer
    DATE CREATED: 19.04.2020
    DATE LAST MODIFIED: 
    PYTHON VERSION: 3.7
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
import json # Import module for label mapping
import datetime # Date and time for time stamp in file name
import pytz # Adjust time zone for time stamp in file name
from PIL import Image # Process image for prediction
import utility_functions as utf # Import module with custom utility functions


def set_up_workspace():
    """
    Run code to setup up the workspace

    Parameters:     None
    Returns:        None
    """
    # Set defaults for the workspace
    warnings.filterwarnings('ignore')

    # Avoid Error #15: Initializing libiomp5.dylib, but found libiomp5.dylib already initialized.
    # https://stackoverflow.com/questions/53014306/error-15-initializing-libiomp5-dylib-but-found-libiomp5-dylib-already-initial
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

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

def load_data_set():
    """
    Load the dataset with TensorFlow Datasets and create a training set, a validation set and a test set
    # https://www.tensorflow.org/datasets/catalog/overview
    # https://www.tensorflow.org/datasets/catalog/oxford_flowers102

    Parameters:     None
    Returns:        dataset:        Data set loaded from hub
                    dataset_info:   Info annotated to data set
                    training_set:   Training data set
                    validation_set: Validation data set
                    test_set:       Test data set
                    num_classes:    Number of classes
                    class_names:    List containing all class names
    """
    # Load the dataset with TensorFlow Datasets.
    dataset, dataset_info = tfds.load('oxford_flowers102', as_supervised=True, shuffle_files=True, with_info=True)

    # Create a training set, a validation set and a test set.
    training_set, validation_set, test_set = dataset['train'], dataset['validation'], dataset['test']

    # Get the number of classes in the dataset from the dataset info.
    num_classes = dataset_info.features['label'].num_classes

    # Load mapping from label to category name
    with open('label_map.json', 'r') as f:
        class_names = json.load(f)

    return dataset, dataset_info, training_set, validation_set, test_set, num_classes, class_names

def create_pipeline(training_set, validation_set, test_set):
    """
    Create a pipeline for the training, validation and testing set

    Parameters:     training_set:   Training data set
                    validation_set: Validation data set
                    test_set:       Test data set
    Returns:        batch_size:     Batch size
                    image_size:     Image dimensions (width, height)
                    training_batches: Batches of training data set
                    validation_batches: Batches of validation data set
                    testing_batches: Batches of test data set
    """
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

    return batch_size, image_size, training_batches, validation_batches, testing_batches

def image_generator_pipeline():
    """
    Create pipeline using image generator. See Introduction into Machine Learning with TensorFlow, Chapter 3.4, Notebook 7
    Not yet implemented in script
    Will be implemented in next iteration (as soon as images stored locallly)
    Not yet tested
    This code is put here so I don't forget to implement it the next time I use this script

    Parameters:     tbd
    Returns:        tbd
    """
    # Create a pipeline for each set.
    # Define batch size and image size
    batch_size = 64
    image_size = 224

    # Define directories
    base_dir = os.path.join('./')
    train_dir = os.path.join(base_dir, 'train')
    validation_dir = os.path.join(base_dir, 'validation')
    test_dir = os.path.join(base_dir, 'test')

    # Create generator for training set
    # It is common pactice to introduce randomness to the training data set
    image_gen_train = ImageDataGenerator(rescale=1. / 255,
                                         horizontal_flip=True,
                                         width_shift_range=0.2,
                                         height_shift_range=0.2,
                                         rotation_range=20,
                                         zoom_range=0.2,
                                         shear_range=20,
                                         fill_mode='nearest')

    # Create generator for validation set
    # Non randomness is introduced to the validation set except rescaling the image values to be between 0 and 1
    image_gen_val = ImageDataGenerator(rescale=1. / 255)

    # Create generator for test set
    # Non randomness is introduced to the test set except rescaling the image values to be between 0 and 1
    image_gen_test = ImageDataGenerator(rescale=1. / 255)

    # Set the pipeline fot the training set
    train_data_gen = image_gen_train.flow_from_directory(directory=train_dir,
                                                         batch_size=batch_size,
                                                         shuffle=True,
                                                         target_size=[image_size, image_size],
                                                         class_mode='categorical')

    # Set the pipeline fot the validation set
    val_data_gen = image_gen_val.flow_from_directory(directory=validation_dir,
                                                     batch_size=BATCH_SIZE,
                                                     target_size=(IMG_SHAPE, IMG_SHAPE),
                                                     class_mode='binary')

    # Set the pipeline fot the test set
    test_data_gen = image_gen_test.flow_from_directory(directory=test_dir,
                                                     batch_size=BATCH_SIZE,
                                                     target_size=(IMG_SHAPE, IMG_SHAPE),
                                                     class_mode='binary')

def train_classifier(image_size, training_batches, validation_batches, num_classes):
    """
    Build and train your network

    Parameters:     image_size:     Image dimensions (width, height)
                    training_batches: Batches of training data set
                    validation_batches: Batches of validation data set
                    num_classes:    Number of classes
    Returns:        model:          Trained model
                    history:        History / Details on the training of the model
                    save_best_model: Saves best model in time stamped file under path_and_filename

    """
    # URL to the MobileNet pre-trained model
    URL = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4"

    # Load pre-trained MobileNet feature extractor and wrap it into a Keras layer to be used as the first layer in the model
    feature_extractor = hub.KerasLayer(URL, input_shape=(image_size, image_size, 3))

    # Set the feature extractor as untrainable (as it is pre-trained)
    feature_extractor.trainable = False

    # Define dropout rate
    dropout_rate = 0.2

    # Number of epochs. The high number is on purpose, as early stopping is implemented.
    num_max_epochs = 50

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

    return model, history

def main():
    """
    main function

    Parameters:     None
    Returns:        None
    """
    # Set up the workspace
    set_up_workspace()

    # Load dataset and create training set, validation set and test set
    dataset, dataset_info, training_set, validation_set, test_set, num_classes, class_names = load_data_set()

    # Explore loaded data set using utility functions
    utf.explore_dataset(dataset_info, training_set, num_classes, class_names)

    # Create a pipeline for the training, validation and testing set
    batch_size, image_size, training_batches, validation_batches, testing_batches = create_pipeline(training_set, validation_set, test_set)

    # Build and train model
    model, history = train_classifier(image_size, training_batches, validation_batches, num_classes)

    # Plot loss & accuracy for training & validation set using utility functions
    utf.training_performance(history)

    # Print the loss and accuracy values achieved on the entire test set using utility functions
    utf.model_test(model, testing_batches, class_names)

    # Show all matplotlib plots
    plt.show()

# Run main function
if __name__ == '__main__':
    main()