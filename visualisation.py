"""
Used to generate and visualise the decision tree and the data splot
"""

import matplotlib.pyplot as plt
import numpy as np
from node_class import Node

# Function to assign positions to the tree nodes
def assign_positions(tree, x=0, y=0, vertical_dist=1, horizontal_dist=2, positions=None):
    if positions is None:
        positions = {}

    # Assign current node position
    print(tree.attribute)
    positions[tree.val] = (x, y, tree.attribute)

    # Calculate positions for the left and right children
    if tree.left:
        positions = assign_positions(tree.left, x - horizontal_dist, y - vertical_dist, vertical_dist, horizontal_dist / 1.5, positions)
    if tree.right:
        positions = assign_positions(tree.right, x + horizontal_dist, y - vertical_dist, vertical_dist, horizontal_dist / 1.5, positions)

    return positions

# Function to plot the tree using the calculated positions
def plot_tree(tree, positions):
    fig, ax = plt.subplots()

    # Plot the edges
    def plot_edges(tree, positions):
        if tree.left:
            x_values = [positions[tree.val][0], positions[tree.left.val][0]]
            y_values = [positions[tree.val][1], positions[tree.left.val][1]]
            ax.plot(x_values, y_values, 'k-', lw=1)  # black line for edges
            plot_edges(tree.left, positions)
        if tree.right:
            x_values = [positions[tree.val][0], positions[tree.right.val][0]]
            y_values = [positions[tree.val][1], positions[tree.right.val][1]]
            ax.plot(x_values, y_values, 'k-', lw=1)
            plot_edges(tree.right, positions)

    plot_edges(tree, positions)

    # Plot the nodes
    for node_val, (x, y, attribute) in positions.items():
        ax.text(x, y, f"X{attribute} < {node_val}", fontsize=12, ha='center', va='center', bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', edgecolor='black'))

    ax.set_aspect('equal')
    ax.set_axis_off()
    plt.savefig("tree.png")



if __name__ == "__main__":
    root = Node()
    root.attribute = "X1"
    root.val = 1.5

    # Left child of root
    root.left = Node()
    root.left.attribute = "X2"
    root.left.val = 2.5

    # Right child of root
    root.right = Node()
    root.right.attribute = "X3"
    root.right.val = 3.5

    # Left child of the left node (leaf node)
    root.left.left = Node()
    root.left.left.attribute = "X4"
    root.left.left.val = 4.5
    root.left.left.leaf = True  # Mark this as a leaf node

    # Right child of the left node (leaf node)
    root.left.right = Node()
    root.left.right.attribute = "X5"
    root.left.right.val = 5.5
    root.left.right.leaf = True  # Mark this as a leaf node

    # Right child of the right node (leaf node)
    root.right.right = Node()
    root.right.right.attribute = "X6"
    root.right.right.val = 6.5
    root.right.right.leaf = True  # Mark this as a leaf node


    # # Assign positions to the tree nodes
    positions = assign_positions(root)

    # # Plot the tree
    plot_tree(root, positions)
