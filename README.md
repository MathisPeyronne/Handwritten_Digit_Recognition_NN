# Handwritten_Digit_Recognition_NN
A fully connected Neural Network for handwritten digit recognition.

## Network Architecture

- 3-layer NN, 500+300 hidden units
- Activation function: ReLU for hidden layers and softmax for the output layer
- Cost function: cross entropy loss function
- Backpropagation: AdamOptimizer

#### Hyperparameters used

- learning rate: 0.0001
- epochs: 300
- batch size: 128

## Dataset

The network is trained on [MNIST](http://yann.lecun.com/exdb/mnist/) handwritten digits. The dataset contains 60'000 examples for training and 10'000 examples for testing. the handwritten digits come as 28x28 fixed-size images with values ranging from 0 to 1. Each image is then flattened to a 1-D numpy array of 784 features.

## Results

- Training Accuracy: 1.0
- Testing Accuracy: 0.9822

## Details

- It uses tensorflow.
- To train: ```python MNIST_NN.py```
- A tensorflow summary is included with the evolution of the weights, biases ,and costs.
- To access tensorboard: ```tensorboard --logdir=data/logs```
