# Image-Classifier-MobileNet-V2

## Project Description
MobileNet CNN Image classifier build on TensorFlow. Developed as a project as partial fulfillment of Udacity Nanodegree "Introduction into Machine Learning with TensorFlow".

1) train.py defines a CNN based on MobileNet-V2 in TensorFlow, trained and results in a model with about 80% accuracy.
2) predict.py rebuilds the model and predict the class of an input image

## Conclusions
- Implementing Tensorflow felt very straight forward.
- Training the MobileNet-V2 for this application was surprisingly quick (only classifier trainied).
- The performance of the model is surprisingly good.

## Project Files
- README.md: Project README
- train.py: Define and train the CNN
- label_map.json: json file containing the categories (predicted classes)
- predict.py: Rebuild the model and predict the class of an input image
- utility_functions: Utility functions for train.py and predict.py


## Data Set
https://www.tensorflow.org/datasets/catalog/overview

https://www.tensorflow.org/datasets/catalog/oxford_flowers102


## Model
The model is based on MobileNet-V2. The pretrained feature extractor is used without modifications.

The custom classifier is used instead of the pretrained classifier. The classifier has following structure:
- The hidden layers have 512 and 256 nodes respectively and use ReLU activation function
- There are dropout layers with 20% dropout rate between the hidden layers.
- The output layer has 102 nodes, as there are 102 classes, and uses Softmax activation function.


## Python Version & IDE
- Python Version: 3.7
- IDE: PyCharm

## Python Modules
- warnings
- numpy
- matplotlib.pyplot
- tensorflow
- tensorflow_datasets
- os
- tensorflow.keras.preprocessing.image import ImageDataGenerator
- tensorflow_hub
- logging
- json
- datetime
- pytz
- PIL


## Bugs
No known bugs

## Pending Tasks
Create an image pipeline using ImageDataGenerator. This will help reduce overfitting of the model, i.e., the validation accuracy will better follow the training accuracy.


## MIT License

Copyright (c) 2018 Udacity

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

