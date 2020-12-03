import numpy as np


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


def softmax_with_cross_entropy(predictions, target_index):
    '''
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
    '''
    # TODO implement softmax with cross-entropy
    # Your final implementation shouldn't have any loops
    probs = softmax(predictions)
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


def l2_regularization(W, reg_strength):
    '''
    Computes L2 regularization loss on weights and its gradient

    Arguments:
      W, np array - weights
      reg_strength - float value

    Returns:
      loss, single value - l2 regularization loss
      gradient, np.array same shape as W - gradient of weight by l2 loss
    '''

    # TODO: implement l2 regularization and gradient
    # Your final implementation shouldn't have any loops
    loss = reg_strength * np.linalg.norm(W)**2
    grad = 2 * reg_strength * W

    return loss, grad
    

def linear_softmax(X, W, target_index):
    '''
    Performs linear classification and returns loss and gradient over W

    Arguments:
      X, np array, shape (num_batch, num_features) - batch of images
      W, np array, shape (num_features, classes) - weights
      target_index, np array, shape (num_batch) - index of target classes

    Returns:
      loss, single value - cross-entropy loss
      gradient, np.array same shape as W - gradient of weight by loss

    '''
    predictions = np.dot(X, W)

    # TODO implement prediction and gradient over W
    # Your final implementation shouldn't have any loops
    loss, dprediction = softmax_with_cross_entropy(predictions, target_index)
    #gradient = np.einsum('ik, jl', X, np.eye(W.shape[1], W.shape[1]))
    #dW = np.einsum('ijkl, ij', gradient, dprediction)
    dW = np.dot(X.T, dprediction)
    
    
    return loss, dW


class LinearSoftmaxClassifier():
    def __init__(self):
        self.W = None

    def fit(self, X, y, batch_size=100, learning_rate=1e-7, reg=1e-5,
            epochs=1, tol=1e-5):
        '''
        Trains linear classifier
        
        Arguments:
          X, np array (num_samples, num_features) - training data
          y, np array of int (num_samples) - labels
          batch_size, int - batch size to use
          learning_rate, float - learning rate for gradient descent
          reg, float - L2 regularization strength
          epochs, int - number of epochs
        '''

        num_train = X.shape[0]
        num_features = X.shape[1]
        num_classes = np.max(y)+1
        if self.W is None:
            self.W = 0.001 * np.random.randn(num_features, num_classes)

        loss_history = []
        for epoch in range(epochs):
            shuffled_indices = np.arange(num_train)
            np.random.shuffle(shuffled_indices)
            sections = np.arange(batch_size, num_train, batch_size)
            batches_indices = np.array_split(shuffled_indices, sections)

            # TODO implement generating batches from indices
            # Compute loss and gradients
            # Apply gradient to weights using learning rate
            # Don't forget to add both cross-entropy loss
            # and regularization!
            batches_X = [X[idx] for idx in batches_indices]
            batches_y = [y[idx] for idx in batches_indices]
            loss = 0
            for batch_X, batch_y in zip(batches_X, batches_y):
                loss_smax, dW = linear_softmax(batch_X, self.W, batch_y)
                loss_reg, dreg = l2_regularization(self.W, reg)
                loss += loss_smax + loss_reg
                self.W -= learning_rate * (dW + dreg)
            loss_history.append(loss)
            if len(loss_history) > 1 and np.abs(loss_history[-2] - loss_history[-1]) < tol:
                break
            # end
            #print("Epoch %i, loss: %f" % (epoch, loss))

        return loss_history

    def predict(self, X):
        '''
        Produces classifier predictions on the set
       
        Arguments:
          X, np array (test_samples, num_features)

        Returns:
          y_pred, np.array of int (test_samples)
        '''
        y_pred = np.zeros(X.shape[0], dtype=np.int)

        # TODO Implement class prediction
        # Your final implementation shouldn't have any loops
        y_pred = np.argmax(np.dot(X, self.W), axis=1)

        return y_pred



                
                                                          

            

                
