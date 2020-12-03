import numpy as np

from layers import (
    FullyConnectedLayer, ReLULayer,
    ConvolutionalLayer, MaxPoolingLayer, Flattener,
    softmax_with_cross_entropy, l2_regularization
    )


class ConvNet:
    """
    Implements a very simple conv net

    Input -> Conv[3x3] -> Relu -> Maxpool[4x4] ->
    Conv[3x3] -> Relu -> MaxPool[4x4] ->
    Flatten -> FC -> Softmax
    """
    def __init__(self, input_shape, n_output_classes, conv1_channels, conv2_channels):
        """
        Initializes the neural network

        Arguments:
        input_shape, tuple of 3 ints - image_width, image_height, n_channels
                                         Will be equal to (32, 32, 3)
        n_output_classes, int - number of classes to predict
        conv1_channels, int - number of filters in the 1st conv layer
        conv2_channels, int - number of filters in the 2nd conv layer
        """
        # TODO Create necessary layers
        input_width, input_height, input_channels = input_shape
        self.conv1 = ConvolutionalLayer(input_channels, conv1_channels, filter_size=3, padding=1)
        self.relu1 = ReLULayer()
        self.maxpool1 = MaxPoolingLayer(pool_size=4, stride=4)
        self.conv2 = ConvolutionalLayer(conv1_channels, conv2_channels, filter_size=3, padding=1)
        self.relu2 = ReLULayer()
        self.maxpool2 = MaxPoolingLayer(pool_size=4, stride=4)
        self.flattener = Flattener()
        self.fc = FullyConnectedLayer(input_width*input_height*conv2_channels // (4**4), n_output_classes)

    def compute_loss_and_gradients(self, X, y):
        """
        Computes total loss and updates parameter gradients
        on a batch of training examples

        Arguments:
        X, np array (batch_size, height, width, input_features) - input data
        y, np array of int (batch_size) - classes
        """
        # Before running forward and backward pass through the model,
        # clear parameter gradients aggregated from the previous pass
        for layer in (self.conv1, 
                      self.conv2, 
                      self.fc):
            for param in layer.params().values():
                param.zero_grad()

        # TODO Compute loss and fill param gradients
        # Don't worry about implementing L2 regularization, we will not
        # need it in this assignment
        conv1_fwd = self.conv1.forward(X)
        relu1_fwd = self.relu1.forward(conv1_fwd)
        maxpool1_fwd = self.maxpool1.forward(relu1_fwd)
        conv2_fwd = self.conv2.forward(maxpool1_fwd)
        relu2_fwd = self.relu2.forward(conv2_fwd)
        maxpool2_fwd = self.maxpool2.forward(relu2_fwd)
        flattener_fwd = self.flattener.forward(maxpool2_fwd)
        fc_fwd = self.fc.forward(flattener_fwd)
        
        loss, dprediction = softmax_with_cross_entropy(fc_fwd, y)
        
        fc_bwd = self.fc.backward(dprediction)
        flattener_bwd = self.flattener.backward(fc_bwd)
        maxpool2_bwd = self.maxpool2.backward(flattener_bwd)
        relu2_bwd = self.relu2.backward(maxpool2_bwd)
        conv2_bwd = self.conv2.backward(relu2_bwd)
        maxpool1_bwd = self.maxpool1.backward(conv2_bwd)
        relu1_bwd = self.relu1.backward(maxpool1_bwd)
        conv1_bwd = self.conv1.backward(relu1_bwd)
        
        return loss

    def predict(self, X):
        # You can probably copy the code from previous assignment
        pred = np.zeros(X.shape[0], np.int)

        conv1_fwd = self.conv1.forward(X)
        relu1_fwd = self.relu1.forward(conv1_fwd)
        maxpool1_fwd = self.maxpool1.forward(relu1_fwd)
        conv2_fwd = self.conv2.forward(maxpool1_fwd)
        relu2_fwd = self.relu2.forward(conv2_fwd)
        maxpool2_fwd = self.maxpool2.forward(relu2_fwd)
        flattener_fwd = self.flattener.forward(maxpool2_fwd)
        fc_fwd = self.fc.forward(flattener_fwd)
        
        pred = np.argmax(fc_fwd, axis=1)
        return pred

    def params(self):
        result = {}

        # TODO: Aggregate all the params from all the layers
        # which have parameters
        for layer_name, layer in (('conv1', self.conv1), 
                                  ('conv2', self.conv2), 
                                  ('fc', self.fc)):
            params = layer.params()
            for param_name in params:
                result[f'{layer_name}.{param_name}'] = params[param_name]

        return result
