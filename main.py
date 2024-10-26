import numpy as np
import func
import visualisation as vis
import evaluation as eval
from numpy.random import default_rng

def main():
    """
    Main function
    """
    dataset = func.parse("noisy_dataset")

    # Split the dataset into training and test sets
    x_train, x_test, y_train, y_test = eval.split_dataset(dataset[:-1], dataset[-1], 0.2)

    train_dataset = list(zip(x_train, y_train))

    # only feed the learning dataset the training datset - NOT THE TEST DATASET
    tree, depth = func.decision_tree_learning(train_dataset, 1)
    # func.troubleshoot(tree)
    positions, node_ids = vis.assign_positions(tree)


    # do some evaluation here
    

    # Plot the tree using the calculated positions and node IDs
    vis.plot_tree(tree, positions, node_ids)

main()