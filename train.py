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
# from IPython import get_ipython, set_matplotlib_formats # Import module to enable IPython magic commands such as %matplotlib inline and %config InlineBackend.figure_format = 'retina'
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
    # get_ipython().run_line_magic('matplotlib', 'inline')
    # set_matplotlib_formats('retina')

    # Disable progress bar to keep shell nice and tidy
    tfds.disable_progress_bar()

    # Define logger
    logger = tf.get_logger()
    logger.setLevel(logging.ERROR)

    # Print some details
    print('Using:')
    print('\t\u2022 TensorFlow version:', tf.__version__)
    print('\t\u2022 tf.keras version:', tf.keras.__version__)
    print('\t\u2022 Running on GPU' if tf.test.is_gpu_available() else '\t\u2022 GPU device not found. Running on CPU')

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

def explore_dataset():
    """
    Explore the loaded dada set
    """
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

    # Indicate and of script
    print('End of script')


# Run main function
if __name__ == '__main__':
    main()