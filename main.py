import numpy as np
import func
import visualisation as vis
import evaluation as eval
import samuel_evaluation as evalualtion
from numpy.random import default_rng

def main():
    """
    Main function
    """
    # Step 1: Parse the dataset
    x, y = func.parse("clean_dataset")

    # Step 2: Perform K-fold evaluation on the data
    data = np.column_stack((x, y))  # Combine features and labels into one dataset
    avg_classification_rate, avg_recall, avg_precision, avg_F1, avg_confusion_matrix = evalualtion.K_fold_evaluation(data)
    
    # Print cross-validation metrics
    print("Cross-Validation Metrics (10-fold):")
    print(f"Average Classification Rate: {avg_classification_rate:.2f}")
    print(f"Average Recall: {avg_recall}")
    print(f"Average Precision: {avg_precision}")
    print(f"Average F1 Score: {avg_F1}")
    print(f"Average Confusion Matrix:\n{avg_confusion_matrix}")

    # Step 3: Split the dataset into training and test sets for visualization
    x_train, x_test, y_train, y_test = eval.split_dataset(x, y, 0.2)
    train_dataset = np.column_stack((x_train, y_train))

    # Save training dataset to file for reference (optional)
    with open('train_dataset.txt', 'w') as file:
        for item in train_dataset:
            file.write(f"{item}\n")

    # Step 4: Train a single tree on the entire training dataset for visualization
    tree, depth = func.decision_tree_learning(train_dataset, 1)
    
    # Step 5: Visualize the tree using `assign_positions` and `plot_tree`
    positions, node_ids = vis.assign_positions(tree)
    vis.plot_tree(tree, positions, node_ids, depth)

# Run the main function
if __name__ == "__main__":
    main()
