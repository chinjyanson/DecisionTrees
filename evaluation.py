'''
10-fold cross-validation of a decision tree with performance evaluation.
'''
import numpy as np
from decision_tree import decision_tree_learning
from node_class import Node
from typing import Dict, Tuple

def K_fold_evaluation(
    data: np.ndarray,
    num_folds: int = 10,
    shuffle: bool = True
) -> Dict[str, np.ndarray]:
    '''
    Performs K-fold cross-validation on a dataset using a decision tree.

    Args:
        data (numpy.ndarray): The dataset to evaluate, where the last column is the label.
        num_folds (int): The number of folds to use in cross-validation.
        shuffle (bool): Whether to shuffle the data before splitting into folds.

    Returns:
        Dict[str, np.ndarray]:
            Dictionary containing the following keys:
            - "classification_rate": Average classification rate as a float.
            - "recall": Average recall per class as a NumPy array.
            - "precision": Average precision per class as a NumPy array.
            - "F1_score": Average F1 score per class as a NumPy array.
            - "confusion_matrix": Average confusion matrix as a NumPy array.
    '''
    if shuffle:
        np.random.shuffle(data)

    # Split data into K folds
    folds = np.array_split(data, num_folds)

    # Initialize lists to collect evaluation metrics across folds
    classification_rates = []
    recall_list, precision_list, F1_list = [], [], []
    confusion_matrices = []

    for index in range(num_folds):
        # Define current fold as test set; the remaining folds as training set
        test_data = folds[index]
        train_data = np.concatenate(folds[:index] + folds[index + 1:])

        # Train a decision tree on the training set
        print('Training decision tree on training folds...')
        tree, _ = decision_tree_learning(train_data, 0)

        # Evaluate the tree on the test set
        conf_matrix, recall, precision, F1, classification_rate = evaluate(test_data, tree)
        print('-'*50)

        # Store metrics for the current fold
        classification_rates.append(classification_rate)
        recall_list.append(recall)
        precision_list.append(precision)
        F1_list.append(F1)
        confusion_matrices.append(conf_matrix)

    # Calculate average evaluation metrics
    avg_classification_rate = np.mean(classification_rates)
    avg_recall = np.mean(recall_list, axis=0)
    avg_precision = np.mean(precision_list, axis=0)
    avg_F1 = np.mean(F1_list, axis=0)
    avg_confusion_matrix = np.mean(confusion_matrices, axis=0)

    # Return metrics as a dictionary for clarity
    return {
        'classification_rate': avg_classification_rate,
        'recall': avg_recall,
        'precision': avg_precision,
        'F1_score': avg_F1,
        'confusion_matrix': avg_confusion_matrix
    }


def evaluate(test_data: np.ndarray, tree: Node) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    '''
    Evaluates a trained decision tree on a test dataset.

    Args:
        test_data (numpy.ndarray): Test dataset with features and labels.
        tree (Node): The root node of the trained decision tree.

    Returns:
        tuple: Evaluation metrics including:
            - confusion_matrix (numpy.ndarray): Confusion matrix (num_classes x num_classes).
            - recall (numpy.ndarray): Per-class recall values.
            - precision (numpy.ndarray): Per-class precision values.
            - F1 (numpy.ndarray): Per-class F1 scores.
            - classification_rate (float): Overall accuracy of the tree on the test set.
    '''
    num_classes = 4
    conf_matrix = np.zeros((num_classes, num_classes), dtype=int)

    # Predict class for each sample in the test set
    for sample in test_data:
        node = tree
        features = sample[:-1]
        true_label = int(sample[-1])

        # Traverse tree to make a prediction
        while not node.leaf:
            if features[node.attribute - 1] <= node.val:
                node = node.left
            else:
                node = node.right

        # Get predicted label at the leaf node and update confusion matrix
        predicted_label = int(node.val)
        conf_matrix[true_label - 1, predicted_label - 1] += 1

    # Calculate recall, precision, F1, and classification rate
    recall, precision, F1, classification_rate = calculate_metrics(conf_matrix)
    return conf_matrix, recall, precision, F1, classification_rate


def calculate_metrics(conf_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    '''
    Calculate recall, precision, F1 score, and classification rate from a confusion matrix.

    Args:
        conf_matrix (numpy.ndarray): Confusion matrix (num_classes x num_classes).

    Returns:
        tuple: Metrics derived from the confusion matrix:
            - recall (numpy.ndarray): Per-class recall values.
            - precision (numpy.ndarray): Per-class precision values.
            - F1 (numpy.ndarray): Per-class F1 scores.
            - classification_rate (float): Overall accuracy.
    '''
    num_classes = conf_matrix.shape[0]
    recall = np.zeros(num_classes)
    precision = np.zeros(num_classes)
    F1 = np.zeros(num_classes)

    for label in range(num_classes):
        true_positive = conf_matrix[label, label]
        actual_positive = np.sum(conf_matrix[label, :])
        predicted_positive = np.sum(conf_matrix[:, label])

        # Calculate recall, precision, and F1 score
        recall[label] = true_positive / actual_positive if actual_positive > 0 else 0
        precision[label] = true_positive / predicted_positive if predicted_positive > 0 else 0
        F1[label] = 2 * (precision[label] * recall[label]) / (precision[label] + recall[label]) if (precision[label] + recall[label]) > 0 else 0

    # Classification rate: ratio of correct predictions to total predictions
    classification_rate = np.trace(conf_matrix) / np.sum(conf_matrix)
    return recall, precision, F1, classification_rate
