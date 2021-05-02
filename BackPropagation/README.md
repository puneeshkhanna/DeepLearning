## Problem Statement

In this assignment, you will build a complete neural network using Numpy.
You will implement all the steps required to build a network - feedforward, loss computation, backpropagation, weight updates etc.

You will use the MNIST dataset to train your model to classify handwritten digits between 0-9.

The assignment is divided into the following sections:

Data preparation
Feedforward
Loss computation
Backpropagation
Parameter updates
Model training and predictions

## Data Preparation
Firstly, we load the data using the function load_data(). 
The function data_wrapper() is then applied to the data to get the train and test data in the desired shape.
Please note that the code needs to take a batch of data points as the input. Hence, be careful while checking the dimensions.
You already know that we have 28x28 greyscale images in the MNIST dataset. Hence, each input image is a vector of length 784. 
The ground truth labels of a batch are stored in a matrix which is converted to a one-hot matrix. Also, the output of the model is a softmax output of length 10.

Hence, we have the following:

train_set_x shape: (784, 50000)
train_set_y shape: (10, 50000)
test_set_x shape: (784, 10000)
test_set_y shape: (10, 10000)

## Feedforward
The whole data is taken as one batch. No minibatch gradient descent is performed.

## Loss Calculation
The loss used for multiclass classification is the cross-entropy loss. 

## Backpropagation
The parameters dictionary is getting updated in place at each step.
The memories from L_layer_forward consisting of the tuple memory = (linear_memory, activation_memory) for each layer is used in backpropagation
The backpropagation process will run in a loop from the last layer to the first, and each loop will compute the gradient.
