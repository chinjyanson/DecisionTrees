import numpy as np
import decision_tree
import visualisation as vs
import evaluation as ev

def main():
    """
    Main function
    """
    # Step 1: Parse the dataset
    dataset_name = input("Please enter the name of the dataset file without the .txt suffix: ")
    x, y = decision_tree.parse(dataset_name)
    data = np.column_stack((x, y)) # Combine features and labels into one dataset

    # Step 2: Visualise the whole dataset as a decision tree
    tree, depth = decision_tree.decision_tree_learning(data, 0)
    vs.visualise(tree, depth)

   # Step 3: Perform K-fold evaluation on the data
    results = ev.K_fold_evaluation(data)

    # Print cross-validation metrics in a structured format
    print("\nCross-Validation Metrics (10-fold):")
    print(f"Average Classification Rate: {results['classification_rate']:.2f}")
    print(f"Average Recall per Class: {results['recall']}")
    print(f"Average Precision per Class: {results['precision']}")
    print(f"Average F1 Score per Class: {results['F1_score']}")
    print(f"Average Confusion Matrix:\n{results['confusion_matrix']}")


# Run the main function
if __name__ == "__main__":
    main()
