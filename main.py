import numpy as np
import func
import visualisation as vis
import evaluation as eval
from numpy.random import default_rng

def main():
    """
    Main function
    """
    # Step 1: Parse the dataset
    x, y = func.parse("clean_dataset")
    data = np.column_stack((x, y)) # Combine features and labels into one dataset

    # Step 2: Visualise the whole dataset as a decision tree
    tree, depth = func.decision_tree_learning(data, 0)
    vis.visualise(tree, depth)

    # Step 3: Perform K-fold evaluation on the data
    avg_classification_rate, avg_recall, avg_precision, avg_F1, avg_confusion_matrix = eval.K_fold_evaluation(data)
    
    # Print cross-validation metrics
    print("Cross-Validation Metrics (10-fold):")
    print(f"Average Classification Rate: {avg_classification_rate:.2f}")
    print(f"Average Recall: {avg_recall}")
    print(f"Average Precision: {avg_precision}")
    print(f"Average F1 Score: {avg_F1}")
    print(f"Average Confusion Matrix:\n{avg_confusion_matrix}")

# Run the main function
if __name__ == "__main__":
    main()
