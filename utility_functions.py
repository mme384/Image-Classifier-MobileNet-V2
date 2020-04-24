#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
    FILE NAME: utility_functions
    AUTHOR: Michalis Meyer
    DATE CREATED: 21.04.2020
    DATE LAST MODIFIED: 
    PYTHON VERSION: 3.7
    SCRIPT PURPOSE: Utility functions
"""
# Import modules
import numpy as np # Import module to use numpy
import matplotlib.pyplot as plt # Import module to use matplotlib
import tensorflow as tf # Import module to use tensorflow

def explore_dataset(dataset_info, training_set, num_classes, class_names):
    """
    Explore the loaded dada set
    (Training utility function)

    Parameters:     dataset_info:   Info annotated to data set
                    training_set:   Training data set
                    num_classes:    Number of classes
                    class_names:    List containing all class names
    Returns:        Prints information on the data set
                    Plots a sample image
    """
    # Display dataset_info
    dataset_info

    # Get the number of examples in each set from the dataset info.
    train_num_examples = dataset_info.splits['train'].num_examples
    validation_num_examples = dataset_info.splits['validation'].num_examples
    test_num_examples = dataset_info.splits['test'].num_examples

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
        # plt.show()

def training_performance(history):
    """
    Plot the loss and accuracy values achieved during training for the training and validation set
    (Training utility function)

    Parameters:     history:        History / Details on the training of the model
    Returns:        Plots the training and validation accuracy
                    Plots the training and validation loss
    """
    # Get training and validation accuracy
    training_accuracy = history.history['accuracy']
    validation_accuracy = history.history['val_accuracy']

    # Get training and validation loss
    training_loss = history.history['loss']
    validation_loss = history.history['val_loss']

    # Determine how many epochs the model was trained
    epochs_range = range(len(training_accuracy))

    # Plot training and validation accuracy and loss
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, training_accuracy, label='Training Accuracy')
    plt.plot(epochs_range, validation_accuracy, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, training_loss, label='Training Loss')
    plt.plot(epochs_range, validation_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    # plt.show()

def model_test(model, testing_batches, class_names):
    """
    Print the loss and accuracy values achieved on the entire test set
    (Training utility function)

    Parameters:     model:          TensorFlow Keras model
                    testing_batches: Batch of test images
                    class_names:    Complete list containing class names and their indices
    Returns:        Plot 30 images from the test batch with correct label, while green labels are correctly classified
                    images and red labels are misclassified images
    """
    # Evaluate the model based on the testing batch
    test_loss, test_accuracy = model.evaluate(testing_batches)

    # Print test accuracy and loss
    print('\nAccuracy on the test set: {:.2%}'.format(test_accuracy))
    print('Loss on test set: {:.2f}'.format(test_loss))

    # Plot some images and their prediction
    # take some images from the testing batch and predict the label
    for image_batch, label_batch in testing_batches.take(1):
        ps = model.predict(image_batch)
        images = image_batch.numpy().squeeze()
        labels = label_batch.numpy()

    # Prepare the plot
    plt.figure(figsize=(10, 15))

    # Plot the images and their prediction. Green titles correctly classified, red titlels misclassified
    print('\nSome images and their prediction. \nGreen titles correctly classified, red titlels misclassified')
    for n in range(30):
        plt.subplot(6, 5, n + 1)
        plt.imshow(images[n], cmap=plt.cm.binary)
        color = 'green' if np.argmax(ps[n]) == labels[n] else 'red'
        plt.title(class_names[str(labels[n])], color=color)
        plt.axis('off')
        # plt.show()

def process_image(image):
    '''
    Process image to be ready for model
    (Prediction utility function)

    Parameters:     Image to be processed
    Returns:        Processed image ready for prediction
    '''
    # Define image size
    image_size = 224

    image = tf.cast(image, tf.float32)
    image = tf.image.resize(image, (image_size, image_size))
    image /= 255
    return image

def show_image(image, probs, classes):
    '''
    Plot figure with a subplot containing the classified image and a subplot containing a vertical bar graph of the
    probabilities of the top k classes predicted by the model
    (Prediction utility function)

    Parameters:     image:          Classified image
                    probs:          Numpy array of top k probabilities predicted by the model
                    classes:        List of top k classes predicted by the model
    Returns:        None
    '''
    fig, (ax1, ax2) = plt.subplots(figsize = (10,5), ncols = 2)
    ax1.imshow(image, cmap = plt.cm.binary)
    ax1.axis('off')
    ax1.set_title('Image')
    ax2.barh(np.arange(probs.size), probs)
    ax2.set_aspect(0.2)
    ax2.set_yticks(np.arange(probs.size))
    ax2.set_yticklabels(classes, size = 'small')
    ax2.set_title('Top k Probabilities')
    ax2.set_xlim(0, 1.1)
    plt.tight_layout()