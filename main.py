import numpy as np
import func
import visualisation as vis
import evaluation as eval
from numpy.random import default_rng

def main():
    """
    Main function
    """
    x, y = func.parse("noisy_dataset")

    # Split the dataset into training and test sets
    x_train, x_test, y_train, y_test = eval.split_dataset(x, y, 0.2)

    train_dataset = np.column_stack((x_train, y_train))

    with open('train_dataset.txt', 'w') as file:
        for item in train_dataset:
            file.write(f"{item}\n")


    # only feed the learning dataset the training datset - NOT THE TEST DATASET
    tree, depth = func.decision_tree_learning(train_dataset, 1)
    #func.troubleshoot(tree)
    positions, node_ids = vis.assign_positions(tree)

    # do some evaluation here
    
    # Plot the tree using the calculated positions and node IDs
    vis.plot_tree(tree, positions, node_ids, depth)

main()