import numpy as np

def binary_classification_metrics(prediction, ground_truth):
    precision = 0
    recall = 0
    accuracy = 0
    f1 = 0

    # TODO: implement metrics!
    
    accuracy = np.count_nonzero(ground_truth == prediction) / len(ground_truth)
    tp = np.count_nonzero((prediction == 1) & (ground_truth == 1))
    fp = np.count_nonzero((prediction == 1) & (ground_truth == 0))
    fn = np.count_nonzero((prediction == 0) & (ground_truth == 1))
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * (precision * recall) / (precision + recall)
    
    return accuracy, precision, recall, f1


def multiclass_accuracy(prediction, ground_truth):
    accuracy = np.count_nonzero(ground_truth == prediction) / len(ground_truth)
    return accuracy
