import numpy as np


def l2_regularization(W, reg_strength):
    """
    Computes L2 regularization loss on weights and its gradient

    Arguments:
      W, np array - weights
      reg_strength - float value

    Returns:
      loss, single value - l2 regularization loss
      gradient, np.array same shape as W - gradient of weight by l2 loss
    """
    # TODO: Copy from the previous assignment
    loss = reg_strength * np.sum(W**2)
    grad = 2 * reg_strength * W
    return loss, grad


def softmax(predictions):
    '''
    Computes probabilities from scores

    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output

    Returns:
      probs, np array of the same shape as predictions - 
        probability for every class, 0..1
    '''
    # TODO implement softmax
    # Your final implementation shouldn't have any loops
    z = np.copy(predictions)
    if len(z.shape) == 1:
        z -= np.max(z)
        ez = np.exp(z)
        ez /= np.sum(ez)
    else:
        z -= np.max(z, axis=1)[:, None]
        ez = np.exp(z)
        ez /= np.sum(ez, axis=1)[:, None]
    return ez

def extract_from_rows(a, col_index):
    if len(a.shape) == 1:
        return a[col_index]
    else:
        return a[np.arange(a.shape[0]), col_index.reshape(-1)]

def cross_entropy_loss(probs, target_index):
    '''
    Computes cross-entropy loss

    Arguments:
      probs, np array, shape is either (N) or (batch_size, N) -
        probabilities for every class
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)

    Returns:
      loss: single value
    '''
    # TODO implement cross-entropy
    # Your final implementation shouldn't have any loops
    return np.mean(-np.log(extract_from_rows(probs, target_index)))


def softmax_with_cross_entropy(preds, target_index):
    """
    Computes softmax and cross-entropy loss for model predictions,
    including the gradient

    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)

    Returns:
      loss, single value - cross-entropy loss
      dprediction, np array same shape as predictions - gradient of predictions by loss value
    """
    # TODO: Copy from the previous assignment
    probs = softmax(preds)
    loss = cross_entropy_loss(probs, target_index)
    dprediction = probs
    probs_target = extract_from_rows(probs, target_index)
    if len(dprediction.shape) == 1:
        dprediction[target_index] = (-1.0 / probs_target) * (-probs_target * probs_target + probs_target)
    else:
        batch_size = dprediction.shape[0]
        dprediction /= batch_size
        dprediction[np.arange(batch_size), target_index.reshape(-1)] = (-1.0 / probs_target) * (-probs_target * probs_target + probs_target) / batch_size

    return loss, dprediction


class Param:
    '''
    Trainable parameter of the model
    Captures both parameter value and the gradient
    '''
    def __init__(self, value):
        self.value = value
        self.grad = np.zeros_like(value)
    
    def zero_grad(self):
        self.grad = np.zeros_like(self.value)

        
class ReLULayer:
    def __init__(self):
        self.X = None

    def forward(self, X):
        # TODO: Implement forward pass
        # Hint: you'll need to save some information about X
        # to use it later in the backward pass
        self.X = Param(X)
        return np.maximum(X, np.zeros(X.shape, dtype=X.dtype))

    def backward(self, d_out):
        """
        Backward pass

        Arguments:
        d_out, np array (batch_size, num_features) - gradient
           of loss function with respect to output

        Returns:
        d_result: np array (batch_size, num_features) - gradient
          with respect to input
        """
        # TODO: Implement backward pass
        # Your final implementation shouldn't have any loops
        self.X.grad = d_out * (self.X.value > 0).astype(int)
        d_result = self.X.grad
        return d_result

    def params(self):
        # ReLU Doesn't have any parameters
        return {}


class FullyConnectedLayer:
    def __init__(self, n_input, n_output):
        self.W = Param(0.001 * np.random.randn(n_input, n_output))
        self.B = Param(0.001 * np.random.randn(1, n_output))
        self.X = None

    def forward(self, X):
        # TODO: Implement forward pass
        # Your final implementation shouldn't have any loops
        self.X = Param(X)
        return np.dot(X, self.W.value) + self.B.value

    def backward(self, d_out):
        """
        Backward pass
        Computes gradient with respect to input and
        accumulates gradients within self.W and self.B

        Arguments:
        d_out, np array (batch_size, n_output) - gradient
           of loss function with respect to output

        Returns:
        d_result: np array (batch_size, n_input) - gradient
          with respect to input
        """
        # TODO: Implement backward pass
        # Compute both gradient with respect to input
        # and gradients with respect to W and B
        # Add gradients of W and B to their `grad` attribute

        # It should be pretty similar to linear classifier from
        # the previous assignment
        self.X.grad = np.dot(d_out, self.W.value.T)
        self.W.grad += np.dot(self.X.value.T, d_out)
        #self.B.grad = np.dot(np.ones((1, d_out.shape[0]), dtype=d_out.dtype), d_out)
        self.B.grad += np.sum(d_out, axis=0)[None, :]

        d_input = self.X.grad
        return d_input

    def params(self):
        return {'W': self.W, 'B': self.B}

    
class ConvolutionalLayer:
    def __init__(self, in_channels, out_channels,
                 filter_size, padding):
        '''
        Initializes the layer
        
        Arguments:
        in_channels, int - number of input channels
        out_channels, int - number of output channels
        filter_size, int - size of the conv filter
        padding, int - number of 'pixels' to pad on each side
        '''

        self.filter_size = filter_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.W = Param(
            np.random.randn(filter_size, filter_size,
                            in_channels, out_channels)
        )

        self.B = Param(np.random.randn(out_channels) / out_channels)

        self.padding = padding
        
        self.X = None


    def forward(self, X):
        self.X = Param(X)
        
        batch_size, height, width, channels = X.shape

        p = self.padding
        f = self.filter_size

        out_height = height + 2*p - (f-1)
        out_width = width + 2*p - (f-1)

        # TODO: Implement forward pass
        # Hint: setup variables that hold the result
        # and one x/y location at a time in the loop below
        
        # It's ok to use loops for going over width and height
        # but try to avoid having any other loops
        output = np.zeros((batch_size, out_height, out_width, self.out_channels))

        W_flat = np.reshape(self.W.value, (-1, self.out_channels))
        padding_shape = ((0,0), (p,p), (p,p), (0,0))
        X_padded = np.pad(X, padding_shape)

        for y in range(out_height):
            for x in range(out_width):
                window = X_padded[:, x:x+f, y:y+f, :]
                window_flat = np.reshape(window, (batch_size, -1))
                convolution = np.dot(window_flat, W_flat)
                output[:, x, y, :] = convolution + self.B.value.reshape((1, -1))

        return output


    def backward(self, d_out):
        # Hint: Forward pass was reduced to matrix multiply
        # You already know how to backprop through that
        # when you implemented FullyConnectedLayer
        # Just do it the same number of times and accumulate gradients

        batch_size, height, width, in_channels = self.X.value.shape
        _, out_height, out_width, out_channels = d_out.shape

        # TODO: Implement backward pass
        # Same as forward, setup variables of the right shape that
        # aggregate input gradient and fill them for every location
        # of the output

        p = self.padding
        f = self.filter_size

        #height = out_height - 2*p + (f-1)
        #width = out_width - 2*p + (f-1)

        padding_shape = ((0,0), (p,p), (p,p), (0,0))
        X_grad_padded = np.pad(np.zeros(self.X.value.shape), padding_shape)
        X_padded = np.pad(self.X.value, padding_shape)

        W_flat = np.reshape(self.W.value, (-1, self.out_channels))
        W_flat_grad = np.zeros(W_flat.shape)

        # Try to avoid having any other loops here too
        for y in range(out_height):
            for x in range(out_width):
                # TODO: Implement backward pass for specific location
                # Aggregate gradients for both the input and
                # the parameters (W and B)
                window_grad_flat = np.dot(d_out[:, x, y, :], W_flat.T) #(batch_size, out_channels) . (f*f*in_channels, out_channels).T = (batch_size, f*f*in_channels)
                window_grad = window_grad_flat.reshape((batch_size, f, f, in_channels))
                X_grad_padded[:, x:x+f, y:y+f, :] += window_grad
                
                window = X_padded[:, x:x+f, y:y+f, :]
                window_flat = np.reshape(window, (batch_size, -1))
                W_flat_grad += np.dot(window_flat.T, d_out[:, x, y, :]) #(batch_size, f*f*in_channels).T . (batch_size, out_channels) = (f*f*in_channels, out_channels)
        
        if p != 0:
            X_grad = X_grad_padded[:, p:-p, p:-p, :]
        else:
            X_grad = X_grad_padded

        W_grad = W_flat_grad.reshape(self.W.value.shape)
        self.W.grad += W_grad
        self.B.grad += np.sum(d_out, axis=(0, 1, 2))
        
        return X_grad

    def params(self):
        return { 'W': self.W, 'B': self.B}  


class MaxPoolingLayer:
    def __init__(self, pool_size, stride):
        '''
        Initializes the max pool

        Arguments:
        pool_size, int - area to pool
        stride, int - step size between pooling windows
        '''
        self.pool_size = pool_size
        self.stride = stride
        self.X = None
        self.max_masks = None

    def forward(self, X):
        self.X = Param(X)
        
        batch_size, height, width, channels = X.shape
        
        s = self.stride
        p = self.pool_size
        
        if np.mod(height - p, s) != 0:
            raise ValueError('Wrong height')
        if np.mod(width - p, s) != 0:
            raise ValueError('Wrong width')
        out_height = int((height - p)/s) + 1
        out_width = int((width - p)/s) + 1
        
        output = np.zeros((batch_size, out_height, out_width, channels))
        
        self.max_masks_flat = Param(
                            np.zeros((out_height, out_width, batch_size, p*p, channels))
        )
        # TODO: Implement maxpool forward pass
        # Hint: Similarly to Conv layer, loop on
        # output x/y dimension
        for y in range(out_height):
            for x in range(out_width):
                window = X[:, s*x:s*x+p, s*y:s*y+p, :]
                window_flat = np.reshape(window, (batch_size, -1, channels)) #TODO: we should also keep channels and fix the code according to this
                max_mask_flat = (np.max(window_flat, axis=1, keepdims=True) == window_flat).astype(int)
                max_mask_flat = max_mask_flat / np.sum(max_mask_flat, axis=1, keepdims=True)
                self.max_masks_flat.value[x, y, :, :, :] = max_mask_flat
                
                max_value = np.sum(window_flat * max_mask_flat, axis=1)
                output[:, x, y, :] = max_value
        
        return output

    def backward(self, d_out):
        # TODO: Implement maxpool backward pass
        batch_size, height, width, channels = self.X.value.shape
        _, out_height, out_width, _ = d_out.shape
        
        s = self.stride
        p = self.pool_size
        
        X_grad = np.zeros(self.X.value.shape)
        
        for y in range(out_height):
            for x in range(out_width):
                max_mask_flat = self.max_masks_flat.value[x, y, :, :, :]
                
                d_flat = np.reshape(d_out[:, x, y, :], (batch_size, -1, channels))
                
                window_grad_flat = d_flat * max_mask_flat
                window_grad = np.reshape(window_grad_flat, (batch_size, p, p, channels))
                X_grad[:, s*x:s*x+p, s*y:s*y+p, :] += window_grad
        
        return X_grad
        

    def params(self):
        return {}


class Flattener:
    def __init__(self):
        self.X_shape = None

    def forward(self, X):
        batch_size, height, width, channels = X.shape
        self.X_shape = X.shape

        # TODO: Implement forward pass
        # Layer should return array with dimensions
        # [batch_size, height*width*channels]
        return np.reshape(X, (batch_size, -1))

    def backward(self, d_out):
        # TODO: Implement backward pass
        return np.reshape(d_out, self.X_shape)

    def params(self):
        # No params!
        return {}
