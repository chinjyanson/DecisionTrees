'''
10-fold validation of the decision tree
'''
from func import decision_tree_learning
import numpy as np

def K_fold_evaluation(data, nr_of_folds=10, shuffle=True):
    '''
    Perform K-fold cross-validation and calculate average performance metrics.
    Returns: average confusion matrix, recall, precision, F1 score,
             and classification rate across the K-folds.
    '''
    if shuffle:
        np.random.shuffle(data)

    # Split data into K folds
    folds = np.array_split(data, nr_of_folds)

    # Initialize arrays to store evaluation measures for each fold
    recall_matrix, precision_matrix, F1_matrix = [], [], []
    classification_rates, confusion_tensor, trees = [], [], []

    for index in range(nr_of_folds):

        # Separate current fold as test set, others as training set
        test_data_set = folds[index]
        training_data_set = np.concatenate(folds[:index] + folds[index + 1:])

        # Train decision tree on training set
        print('TRAINING TREE ON REMAINING FOLDS...')
        tree, _ = decision_tree_learning(training_data_set, 0)
        trees.append(tree)

        # Evaluate trained tree on the test dataset
        confusion_matrix, recall, precision, F1, classification_rate = evaluate(test_data_set, tree)
        print('-'*70)

        # Store evaluation measures
        confusion_matrix = np.reshape(confusion_matrix, (1, 4, 4))  # Shape for stacking later
        classification_rates.append(classification_rate)

        if index == 0:
            recall_matrix, precision_matrix, F1_matrix = recall, precision, F1
            confusion_tensor = confusion_matrix
        else:
            recall_matrix = np.vstack((recall_matrix, recall))
            precision_matrix = np.vstack((precision_matrix, precision))
            F1_matrix = np.vstack((F1_matrix, F1))
            confusion_tensor = np.vstack((confusion_tensor, confusion_matrix))

    # Calculate average of evaluation metrics
    average_recall = np.mean(recall_matrix, axis=0)
    average_precision = np.mean(precision_matrix, axis=0)
    average_F1 = np.mean(F1_matrix, axis=0)
    average_classification_rate = np.mean(classification_rates)
    average_confusion_matrix = np.mean(confusion_tensor, axis=0)

    return (average_classification_rate, average_recall, average_precision, average_F1, average_confusion_matrix)


def evaluate(test_dataset, trained_tree):
    ''' return: confusion matrix, recall, precision, F1 score, classification rate '''
    confusion_matrix = np.zeros((4,4), dtype=int)

    # Traverse each sample in the test dataset using the trained tree
    for sample in test_dataset:
        current_node = trained_tree
        attribute_values = sample[:-1]
        true_label = int(sample[-1])

        # Traverse the tree until a leaf node is reached
        while not current_node.leaf:
            # Go left or right based on the attribute value
            if attribute_values[current_node.attribute - 1] <= current_node.val:
                current_node = current_node.left
            else:
                current_node = current_node.right

        # At the leaf node, get the predicted label and update confusion matrix
        predicted_label = int(current_node.val)
        confusion_matrix[true_label - 1, predicted_label - 1] += 1

    # Calculate metrics from the confusion matrix
    recall, precision, F1, classification_rate = metrics(confusion_matrix)

    return confusion_matrix, recall, precision, F1, classification_rate

def metrics(confusion_matrix):
    '''
    Calculate recall, precision, F1 score, and classification rate from confusion matrix.
    Returns: arrays of recall, precision, F1 for each class, and overall classification rate.
    '''
    num_classes = confusion_matrix.shape[0]
    recall = np.zeros(num_classes)
    precision = np.zeros(num_classes)
    F1 = np.zeros(num_classes)

    for label in range(num_classes):
        # Recall: True Positives / Actual Positives
        true_positive = confusion_matrix[label, label]
        actual_positive = np.sum(confusion_matrix[label, :])
        if actual_positive > 0:
            recall[label] = true_positive / actual_positive

        # Precision: True Positives / Predicted Positives
        predicted_positive = np.sum(confusion_matrix[:, label])
        if predicted_positive > 0:
            precision[label] = true_positive / predicted_positive

        # F1 Score: Harmonic mean of precision and recall
        if recall[label] > 0 and precision[label] > 0:
            F1[label] = 2 * (precision[label] * recall[label]) / (precision[label] + recall[label])

    # Classification rate: sum of True Positives / total samples
    classification_rate = np.trace(confusion_matrix) / np.sum(confusion_matrix)

    return recall, precision, F1, classification_rate
