import numpy as np

from layers import FullyConnectedLayer, ReLULayer, softmax_with_cross_entropy, l2_regularization


class TwoLayerNet:
    """ Neural network with two fully connected layers """

    def __init__(self, n_input, n_output, hidden_layer_size, reg):
        """
        Initializes the neural network

        Arguments:
        n_input, int - dimension of the model input
        n_output, int - number of classes to predict
        hidden_layer_size, int - number of neurons in the hidden layer
        reg, float - L2 regularization strength
        """
        self.reg = reg
        # TODO Create necessary layers
        self.hidden_layer = FullyConnectedLayer(n_input, hidden_layer_size)
        self.relu_hidden = ReLULayer()
        self.output_layer = FullyConnectedLayer(hidden_layer_size, n_output)
        

    def compute_loss_and_gradients(self, X, y):
        """
        Computes total loss and updates parameter gradients
        on a batch of training examples

        Arguments:
        X, np array (batch_size, input_features) - input data
        y, np array of int (batch_size) - classes
        """
        # Before running forward and backward pass through the model,
        # clear parameter gradients aggregated from the previous pass
        # TODO Set parameter gradient to zeros
        # Hint: using self.params() might be useful!
        for layer in (self.hidden_layer, self.output_layer):
            for param in layer.params().values():
                param.zero_grad()
        
        # TODO Compute loss and fill param gradients
        # by running forward and backward passes through the model
        hidden_fwd = self.hidden_layer.forward(X)
        relu_hidden_fwd = self.relu_hidden.forward(hidden_fwd)
        output_fwd = self.output_layer.forward(relu_hidden_fwd)
        
        loss, dprediction = softmax_with_cross_entropy(output_fwd, y)
        
        output_bwd = self.output_layer.backward(dprediction)
        relu_hidden_bwd = self.relu_hidden.backward(output_bwd)
        hidden_bwd = self.hidden_layer.backward(relu_hidden_bwd)
        
        
        # After that, implement l2 regularization on all params
        # Hint: self.params() is useful again!
        for layer in (self.hidden_layer, self.output_layer):
            W = layer.params()['W']
            loss_reg, grad_reg = l2_regularization(W.value, self.reg)
            loss += loss_reg
            W.grad += grad_reg
            
            #B = layer.params()['B']
            #loss_reg, grad_reg = l2_regularization(B.value, self.reg)
            #loss += loss_reg
            #B.grad += grad_reg

        return loss

    def predict(self, X):
        """
        Produces classifier predictions on the set

        Arguments:
          X, np array (test_samples, num_features)

        Returns:
          y_pred, np.array of int (test_samples)
        """
        # TODO: Implement predict
        # Hint: some of the code of the compute_loss_and_gradients
        # can be reused
        pred = np.zeros(X.shape[0], np.int)

        hidden_fwd = self.hidden_layer.forward(X)
        relu_hidden_fwd = self.relu_hidden.forward(hidden_fwd)
        output_fwd = self.output_layer.forward(relu_hidden_fwd)
        
        pred = np.argmax(output_fwd, axis=1)
        return pred

    def params(self):
        result = {}

        # TODO Implement aggregating all of the params

        for layer_name, layer in (('hidden_layer', self.hidden_layer), 
                                  ('output_layer', self.output_layer)):
            params = layer.params()
            for param_name in params:
                result[f'{layer_name}.{param_name}'] = params[param_name]

        return result
