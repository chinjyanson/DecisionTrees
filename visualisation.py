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

def assign_positions(tree, x=0, y=0, vertical_dist=1, horizontal_dist=4, depth=0, positions=None, max_depth=0, node_ids=None):
    """
    Assign positions to each node dynamically based on the depth of the tree.
    """
    global node_counter  # Access the global node counter
    
    if positions is None:
        positions = {}  # Ensure positions is a dictionary
    
    if node_ids is None:
        node_ids = {}

    if max_depth == 0:
        max_depth = tree_depth(tree)

    # Generate a unique ID for the current node
    node_id = node_counter
    node_counter += 1

    # Store the node's unique ID
    node_ids[tree] = node_id

    # Adjust horizontal distance dynamically based on the depth
    current_horizontal_dist = horizontal_dist * (2 ** (max_depth - depth))

    # Debug: Print the node being processed
    print(f"Processing node {node_id} at depth {depth}, X: {x}, Y: {y}, leaf: {tree.leaf if hasattr(tree, 'leaf') else False}")

    # Check if current node is a leaf
    if tree.leaf:
        # Assign positions for the leaf node
        positions[node_id] = (x, y, 'Leaf', tree.val)  # Store node value in the tuple
        print(f"Assigned leaf node {node_id}: (x={x}, y={y}) with value {tree.val}")
    else:
        # Assign positions for the decision node
        positions[node_id] = (x, y, tree.attribute, tree.val)  # Store node value in the tuple
        print(f"Assigned decision node {node_id}: (x={x}, y={y}) with attribute {tree.attribute} and value {tree.val}")

        # Recursively calculate positions for left and right children
        if tree.left:
            positions, node_ids = assign_positions(tree.left, 
                                                   x - current_horizontal_dist,  # Move left child more to the left
                                                   y - vertical_dist, 
                                                   vertical_dist, 
                                                   horizontal_dist, 
                                                   depth + 1,  # Increment the depth
                                                   positions, 
                                                   max_depth,
                                                   node_ids)
        if tree.right:
            positions, node_ids = assign_positions(tree.right, 
                                                   x + current_horizontal_dist,  # Move right child more to the right
                                                   y - vertical_dist, 
                                                   vertical_dist, 
                                                   horizontal_dist, 
                                                   depth + 1,  # Increment the depth
                                                   positions, 
                                                   max_depth,
                                                   node_ids)

    return positions, node_ids


def plot_tree(tree, positions, node_ids, depth):
    fig, ax = plt.subplots(figsize=(2 *depth, 1 *depth))  # Increased the size of the plot for better spacing

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
            print(f"Plotting leaf node {node_id} at (x={x}, y={y}) with value {node_val}")
            ax.text(x, y, f"leaf: {node_val}", fontsize=12, ha='center', va='center', 
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', edgecolor='black'))
        else:
            print(f"Plotting decision node {node_id} at (x={x}, y={y}) with attribute {attribute} and value {node_val}")
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
