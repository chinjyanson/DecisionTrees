"""
Used to generate and visualise the decision tree and the data plot
"""

import matplotlib.pyplot as plt
import numpy as np
from node_class import Node

# Global counter to assign unique IDs
node_counter = 0

def assign_positions(tree: Node, max_depth: int, x=0, y=0, vertical_dist=1, horizontal_dist=4, depth=0, positions=None, node_ids=None):
    """
    Assign positions to each node dynamically based on the depth of the tree.
    Params: tree: Class Node, max_depth: int, x: int, y: int, vertical_dist: int, horizontal_dist: int, depth: int, positions: hash, node_ids: hash
    """
    global node_counter  # Access the global node counter
    
    if positions is None:
        positions = {}  # Ensure positions is a dictionary
    
    if node_ids is None:
        node_ids = {}

    # Generate a unique ID for the current node
    node_id = node_counter
    node_counter += 1

    # Store the node's unique ID
    node_ids[tree] = node_id

    # Adjust horizontal distance dynamically based on the depth
    current_horizontal_dist = horizontal_dist * (2 ** (max_depth - depth))

    # Check if current node is a leaf
    if tree.leaf:
        # Assign positions for the leaf node
        positions[node_id] = (x, y, 'Leaf', tree.val)  # Store node value in the tuple
    else:
        # Assign positions for the decision node
        positions[node_id] = (x, y, tree.attribute, tree.val)  # Store node value in the tuple
   
        # Recursively calculate positions for left and right children
        if tree.left:
            positions, node_ids = assign_positions(tree.left, 
                                                   max_depth,
                                                   x - current_horizontal_dist,  # Move left child more to the left
                                                   y - vertical_dist, 
                                                   vertical_dist, 
                                                   horizontal_dist, 
                                                   depth + 1,  # Increment the depth
                                                   positions, 
                                                   node_ids)
        if tree.right:
            positions, node_ids = assign_positions(tree.right, 
                                                   max_depth,
                                                   x + current_horizontal_dist,  # Move right child more to the right
                                                   y - vertical_dist, 
                                                   vertical_dist, 
                                                   horizontal_dist, 
                                                   depth + 1,  # Increment the depth
                                                   positions, 
                                                   node_ids)

    return positions, node_ids

def plot_tree(tree: Node, positions: hash, node_ids: hash, depth: int)-> None:
    """
    Plotting text boxes for each node anda also draw edges between nodes
    Params: tree: Class Node, positions: hash, node_ids: hash, depth: int
    """
    fig, ax = plt.subplots(figsize=(5*depth, 2 *depth))  # Increased the size of the plot for better spacing
    # fig, ax = plt.subplots(figsize=(15, 8))
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
            ax.text(x, y, f"leaf: {node_val}", fontsize=12, ha='center', va='center', 
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', edgecolor='black'))
        else:
            ax.text(x, y, f"X{attribute} < {node_val}", fontsize=12, ha='center', va='center', 
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', edgecolor='black'))
    
    ax.set_aspect('auto')  # Change aspect to auto for better vertical spacing
    ax.set_axis_off()
    plt.savefig("WholeDecisionTree.png")

def visualise(tree: Node, depth: int)-> bool:
    """
    Used to wrap the functions to visualise the decision tree in a single function
    Params: Tree: Class Node, depth: int
    """
    positions, node_ids = assign_positions(tree, depth)
    plot_tree(tree, positions, node_ids, depth)
    return True