import numpy as np
import func
import visualisation as vis

def main():
    """
    Main function
    """
    dataset = func.parse("clean_dataset")
    tree, depth = func.decision_tree_learning(dataset, 1)
    # func.troubleshoot(tree)
    positions, node_ids = vis.assign_positions(tree)

    # Plot the tree using the calculated positions and node IDs
    vis.plot_tree(tree, positions, node_ids)

main()