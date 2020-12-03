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
    """
    Trainable parameter of the model
    Captures both parameter value and the gradient
    """

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
