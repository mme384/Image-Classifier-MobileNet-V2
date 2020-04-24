#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
    FILE NAME: predict.py
    AUTHOR: Michalis Meyer
    DATE CREATED: 22.04.2020
    DATE LAST MODIFIED: 
    PYTHON VERSION: 3.7
    SCRIPT PURPOSE: Predict the class of an image
    SAMPLE COMMAND LINE: python predict.py --file_path './test_images/hard-leaved_pocket_orchid.jpg' --model_filename 'model_20200422_223607.h5' --top_k 5  --category_names 'label_map.json'
"""
# Import modules
import warnings # Import module to deal with warnings
from argparse import ArgumentParser # Import module to parse arguments
import numpy as np # Import module to use numpy
import matplotlib.pyplot as plt # Import module to use matplotlib
import tensorflow as tf # Import module to use tensorflow
import tensorflow_datasets as tfds # Import module to use tensorflow datasets
import tensorflow_hub as hub # Import module to import model from tensorflow Hub
import json # Import module for label mapping
import os # Import module to deal with path names used in image data generator
from PIL import Image # Process image for prediction
import utility_functions as utf # Import module with custom utility functions

def set_up_workspace():
    """
    Setup up the workspace

    Parameters:     None
    Returns:        None
    """
    # Avoid Error #15: Initializing libiomp5.dylib, but found libiomp5.dylib already initialized.
    # https://stackoverflow.com/questions/53014306/error-15-initializing-libiomp5-dylib-but-found-libiomp5-dylib-already-initial
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

    # Magic command for inline plotting
    # %matplotlib inline
    # %config InlineBackend.figure_format = 'retina'

def get_input_args():
    """
    Retrieve and parse command line arguments.
    Command Line Arguments:
       - Image file path as --file_path
       - Model path and file name as --model_filename
       - Top k classes to be returned as --top_k
       - Path to json file mapping labels as --category_names
    This function returns these arguments as an ArgumentParser object.
    Parameters:
       None - simply using argparse module to create & store command line arguments
    Returns:
       parse_args() - data structure that stores the command line arguments object.
    """
    # Create Parse using ArgumentParser
    parser = ArgumentParser()

    # Image file path as --file_path
    # Path to test images
    # image_path = './test_images/hard-leaved_pocket_orchid.jpg'
    # image_path = './test_images/cautleya_spicata.jpg'
    # image_path = './test_images/orange_dahlia.jpg'
    # image_path = './test_images/wild_pansy.jpg'
    parser.add_argument("--file_path",
                        type = str,
                        default = './test_images/hard-leaved_pocket_orchid.jpg',
                        help = "Image file path.")

    # Model file name as --model_filename
    parser.add_argument("--model_filename",
                        type = str,
                        default = 'model_20200422_223607.h5',
                        help = "Model path and file name.")

    # Top k classes to be returned as --top_k with default value 5
    parser.add_argument("--top_k",
                        type = int,
                        default = 5,
                        help = "Number of epochs. Default = 5")

    # json file mapping labels as --category_names
    parser.add_argument("--category_names",
                        type = str,
                        default = 'label_map.json',
                        help = "json file mapping labels.")

    return parser.parse_args()

def load_model(model_filename):
    """
    Load the Keras model

    Parameters:     The model file name for the trained TensorFlow Keras model
    Returns:        Returns the model
    """
    # Reload model
    model = tf.keras.models.load_model(model_filename, custom_objects={'KerasLayer': hub.KerasLayer})

    # Display model summary
    model.summary()

    return model

def predict(image_path, model, class_names, top_k):
    '''
    Predicts class of image based on model

    Parameters:     image_path      Path of the image to be classified
                    model:          Name of the trained model
                    class_names:    Complete list containing class names and their indices
                    top_k:          Top k probalibilites and classes to be returned by the function
    Returns:        top_k_probs_np  Numpy array of top k probabilities predicted by the model
                    top_k_classes   List of top k classes predicted by the model
    '''
    # Open image
    image = Image.open(image_path)
    # Conevrt image to numpy array
    image_np = np.asarray(image)
    # Process image to be ready for prediction
    processed_image = utf.process_image(image_np)
    # Expand shape (224, 224, 3) to (1, 224, 224, 3) to represent the batch size.
    expanded_image = np.expand_dims(processed_image, axis=0)

    # Predict class
    probs = model.predict(expanded_image)

    # Get top k probabilities and their index
    top_k_probs, top_k_indices = tf.math.top_k(probs, k=top_k, sorted=True)

    # Convert top k probabilities and their index from tf.Tensor to numpy array and squeeze the shape
    top_k_probs_np = top_k_probs.numpy().squeeze()
    top_k_indices_np = top_k_indices.numpy().squeeze()
    # Convert int to str
    top_k_indices_np_str = np.char.mod('%d', top_k_indices_np)

    # Create top_k_classes list
    top_k_classes = []
    [top_k_classes.append(class_names[label]) for label in top_k_indices_np_str]

    return top_k_probs_np, top_k_classes

def main():
    """
    Main function

    Parameters:     None
    Returns:        None
    """
    # Set up the workspace
    set_up_workspace()

    # Assigns variable in_args to parse_args()
    in_args = get_input_args()

    # Load model
    model = load_model(in_args.model_filename)

    # Load mapping from label to category name
    with open(in_args.category_names, 'r') as f:
        class_names = json.load(f)

    # Load test image, convert to numpy array, process image
    org_image = Image.open(in_args.file_path)
    test_image = np.asarray(org_image)
    test_image = utf.process_image(test_image)

    # Predict class and probability of image
    probs, top_k_classes = predict(in_args.file_path, model, class_names, in_args.top_k)

    # Print classes and probabilities
    utf.display_predictions(probs, top_k_classes)

    # Plot image, classes and probabilities
    utf.show_image(test_image, probs, top_k_classes)

    # Show all matplotlib plots made in the script
    plt.show()

# Run main function
if __name__ == '__main__':
    main()