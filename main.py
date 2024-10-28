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
    seed = 60012
    rg = default_rng(seed)
    x_train, x_test, y_train, y_test = eval.split_dataset(dataset[:, :-1], dataset[:, -1], 0.2, rg)
    # print(x_train)
    # print(y_train)

    train_dataset = list(zip(x_train, y_train))

    # print(train_dataset)

    # only feed the learning dataset the training datset - NOT THE TEST DATASET
    tree, depth = func.decision_tree_learning(dataset, 1)
    print(tree)
    # # func.troubleshoot(tree)
    positions, node_ids = vis.assign_positions(tree)

    # do some evaluation here
    

    # Plot the tree using the calculated positions and node IDs
    vis.plot_tree(tree, positions, node_ids, depth)

main()