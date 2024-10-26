"""
Used to generate and visualise the decision tree and the data plot
"""

import matplotlib.pyplot as plt
import numpy as np
from node_class import Node

# Global counter to assign unique IDs
node_counter = 0

def tree_depth(tree):
    """
    Recursively calculates the maximum depth of the tree.
    """
    if tree is None or tree.leaf:
        return 0
    return 1 + max(tree_depth(tree.left), tree_depth(tree.right))

def assign_positions(tree, x=0, y=0, vertical_dist=3, horizontal_dist=2, depth=0, positions=None, max_depth=0, node_ids=None):
    """
    Assign positions to each node dynamically based on the depth of the tree, using unique node IDs.
    """
    global node_counter  # Access the global node counter
    
    if positions is None:
        positions = {}  # Ensure positions is a dictionary
    
    if node_ids is None:
        node_ids = {}

    if max_depth == 0:
        max_depth = tree_depth(tree)

    # Set a uniform horizontal distance between nodes at the same depth
    horizontal_dist = 2 ** (max_depth - 1)  # Use maximum depth for spacing

    # Generate a unique ID for the current node
    node_id = node_counter
    node_counter += 1

    # Store the node's unique ID
    node_ids[tree] = node_id

    # Check if current node is a leaf
    if tree.leaf:
        # Assign positions for the leaf node
        positions[node_id] = (x, y, 'Leaf', tree.val)  # Store node value in the tuple

        # Add two "empty" leaf children to make it visually consistent
        # Left empty child
        positions[node_counter] = (x - horizontal_dist // (2 ** (depth + 1)), y - vertical_dist, 'Empty', None)
        # Right empty child
        positions[node_counter + 1] = (x + horizontal_dist // (2 ** (depth + 1)), y - vertical_dist, 'Empty', None)
        node_counter += 2  # Increment counter for the fake children
    else:
        # Assign positions for the decision node
        positions[node_id] = (x, y, tree.attribute, tree.val)  # Store node value in the tuple

        # Recursively calculate positions for left and right children
        if tree.left:
            positions, node_ids = assign_positions(tree.left, 
                                                   x - horizontal_dist // (2 ** (depth + 1)), 
                                                   y - vertical_dist, 
                                                   vertical_dist, 
                                                   horizontal_dist, 
                                                   depth + 1,  # Increment the depth
                                                   positions, 
                                                   max_depth,
                                                   node_ids)
        if tree.right:
            positions, node_ids = assign_positions(tree.right, 
                                                   x + horizontal_dist // (2 ** (depth + 1)), 
                                                   y - vertical_dist, 
                                                   vertical_dist, 
                                                   horizontal_dist, 
                                                   depth + 1,  # Increment the depth
                                                   positions, 
                                                   max_depth,
                                                   node_ids)

    return positions, node_ids

# Function to plot the tree using the calculated positions
def plot_tree(tree, positions, node_ids):
    fig, ax = plt.subplots(figsize=(12, 8))  # Increased the size of the plot for better spacing

    # Function to plot edges between nodes
    def plot_edges(tree, positions, node_ids):
        node_id = node_ids[tree]  # Get the unique ID for the current node
        if tree.left:
            left_id = node_ids[tree.left]  # Get the unique ID for the left child
            x_values = [positions[node_id][0], positions[left_id][0]]
            y_values = [positions[node_id][1], positions[left_id][1]]
            ax.plot(x_values, y_values, 'k-', lw=1)  # black line for edges
            plot_edges(tree.left, positions, node_ids)
        if tree.right:
            right_id = node_ids[tree.right]  # Get the unique ID for the right child
            x_values = [positions[node_id][0], positions[right_id][0]]
            y_values = [positions[node_id][1], positions[right_id][1]]
            ax.plot(x_values, y_values, 'k-', lw=1)
            plot_edges(tree.right, positions, node_ids)

    plot_edges(tree, positions, node_ids)

    # Plot the nodes with the correct label
    for node_id, (x, y, attribute, node_val) in positions.items():
        if attribute == 'Leaf':
            ax.text(x, y, f"Leaf: {node_val}", fontsize=12, ha='center', va='center', 
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', edgecolor='black'))
        elif attribute == 'Empty':
            ax.text(x, y, "", fontsize=12, ha='center', va='center', 
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='white'))  # Invisible box
        else:
            ax.text(x, y, f"X{attribute} < {node_val}", fontsize=12, ha='center', va='center', 
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', edgecolor='black'))
    
    ax.set_aspect('auto')  # Change aspect to auto for better vertical spacing
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
    positions, node_ids = assign_positions(root)

    # # Plot the tree
    plot_tree(root, positions, node_ids)
