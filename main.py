import numpy as np
import func
import visualisation as vis

def main():
    """
    Main function
    """
    dataset = func.parse("clean_dataset")
    tree, depth = func.decision_tree_learning(dataset, 0)
    # print(tree)
    # print(depth)
    positions = vis.assign_positions(tree)
    vis.plot_tree(tree, positions)

