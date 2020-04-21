#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
    FILE NAME: train_utility_functions
    AUTHOR: Michalis Meyer
    DATE CREATED: 21.04.2020
    DATE LAST MODIFIED: 
    PYTHON VERSION: 3.7
    SCRIPT PURPOSE: Utility functions
"""
# Import modules
import numpy as np # Import module to use numpy
import matplotlib.pyplot as plt # Import module to use matplotlib

def explore_dataset(dataset_info, training_set, num_classes, class_names):
    """
    Training utility function
    Explore the loaded dada set
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
    Training utility function
    Plot the loss and accuracy values achieved during training for the training and validation set
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
    Training utility function
    Print the loss and accuracy values achieved on the entire test set
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