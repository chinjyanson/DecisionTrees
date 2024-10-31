"""
Used to generate and visualise the decision tree and the data plot
"""

import matplotlib.pyplot as plt
from node_class import Node

# Global counter to assign unique IDs
node_counter = 0

def assign_positions(
    tree: Node,
    max_depth: int,
    x: float = 0,
    y: float = 0,
    vertical_dist: float = 1,
    horizontal_dist: float = 4,
    depth: int = 0,
    positions: dict = None,
    node_ids: dict = None
) -> tuple[dict, dict]:
    """
    Assigns positions to each node in the decision tree dynamically based on its depth.

    This function traverses the decision tree and calculates (x, y) coordinates for each node
    to facilitate visualization. It uses a global counter to assign unique IDs to each node.

    Args:
        tree (Node): The current node of the decision tree being processed.
        max_depth (int): The maximum depth of the tree, used to adjust the positioning.
        x (float): The x-coordinate for the current node.
        y (float): The y-coordinate for the current node.
        vertical_dist (float): The vertical distance between levels in the tree.
        horizontal_dist (float): The horizontal distance between nodes.
        depth (int): The current depth of the tree during traversal.
        positions (dict): A dictionary to store node positions and attributes.
        node_ids (dict): A dictionary to store unique IDs for each node.

    Returns:
        tuple[dict, dict]:
            - A dictionary mapping node IDs to their (x, y) positions and attributes.
            - A dictionary mapping each node to its unique identifier.
    """
    global node_counter  # Access the global node counter

    if positions is None:
        positions = {}  # Initialize positions if not provided

    if node_ids is None:
        node_ids = {}  # Initialize node IDs if not provided

    # Generate a unique ID for the current node
    node_id = node_counter
    node_counter += 1

    # Store the node's unique ID
    node_ids[tree] = node_id

    # Adjust horizontal distance dynamically based on the current depth
    current_horizontal_dist = horizontal_dist * (2 ** (max_depth - depth))

    # Check if the current node is a leaf
    if tree.leaf:
        # Assign positions for the leaf node
        positions[node_id] = (x, y, "Leaf", tree.val)  # Store node value in the tuple
    else:
        # Assign positions for the decision node
        positions[node_id] = (x, y, tree.attribute, tree.val)  # Store attribute and node value

        # Recursively calculate positions for left and right children
        if tree.left:
            positions, node_ids = assign_positions(
                tree.left,
                max_depth,
                x - current_horizontal_dist,  # Move left child more to the left
                y - vertical_dist,             # Move down vertically
                vertical_dist,
                horizontal_dist,
                depth + 1,  # Increment the depth
                positions,
                node_ids,
            )
        if tree.right:
            positions, node_ids = assign_positions(
                tree.right,
                max_depth,
                x + current_horizontal_dist,  # Move right child more to the right
                y - vertical_dist,             # Move down vertically
                vertical_dist,
                horizontal_dist,
                depth + 1,  # Increment the depth
                positions,
                node_ids,
            )

    return positions, node_ids  # Return the updated positions and node IDs

def plot_tree(tree: Node, positions: dict, node_ids: dict, depth: int) -> None:
    """
    Plots the decision tree by drawing text boxes for each node and edges between nodes.

    This function creates a visual representation of the decision tree, displaying the
    structure of nodes and their respective attributes or values.

    Args:
        tree (Node): The root node of the decision tree to be plotted.
        positions (dict): A dictionary mapping each node to its (x, y) coordinates and attributes.
        node_ids (dict): A dictionary mapping each node to a unique identifier for plotting.
        depth (int): The current depth of the tree, which affects the size and spacing of the plot.
    """
    fig, ax = plt.subplots(figsize=(5 * depth, 2 * depth))  # Increased size for better spacing

    # Function to plot edges between nodes
    def plot_edges(node: Node, positions: dict, node_ids: dict) -> None:
        """Recursively plots edges between the current node and its children."""
        node_id = node_ids[node]  # Get the unique ID for the current node
        if node.left:
            left_id = node_ids[node.left]  # Unique ID for the left child
            x_values = [positions[node_id][0], positions[left_id][0]]
            y_values = [positions[node_id][1], positions[left_id][1]]
            ax.plot(x_values, y_values, "k-", lw=1)  # Draw black line for edges
            plot_edges(node.left, positions, node_ids)  # Recursive call for left child
        if node.right:
            right_id = node_ids[node.right]  # Unique ID for the right child
            x_values = [positions[node_id][0], positions[right_id][0]]
            y_values = [positions[node_id][1], positions[right_id][1]]
            ax.plot(x_values, y_values, "k-", lw=1)  # Draw black line for edges
            plot_edges(node.right, positions, node_ids)  # Recursive call for right child

    plot_edges(tree, positions, node_ids)  # Initial call to plot edges

    # Plot the nodes with the correct label
    for node_id, (x, y, attribute, node_val) in positions.items():
        if attribute == "Leaf":
            ax.text(
                x,
                y,
                f"leaf: {node_val}",
                fontsize=12,
                ha="center",
                va="center",
                bbox=dict(
                    boxstyle="round,pad=0.3", facecolor="lightgreen", edgecolor="black"
                ),
            )
        else:
            ax.text(
                x,
                y,
                f"X{attribute} < {node_val}",
                fontsize=12,
                ha="center",
                va="center",
                bbox=dict(
                    boxstyle="round,pad=0.3", facecolor="lightblue", edgecolor="black"
                ),
            )

    ax.set_aspect("auto")  # Set aspect to auto for better vertical spacing
    ax.set_axis_off()  # Hide the axis for a cleaner look
    plt.savefig("WholeDecisionTree.png")  # Save the plot as an image

def visualise(tree: Node, depth: int) -> None:
    """
    Visualizes the decision tree by plotting it.

    This function wraps the necessary functions to assign positions
    to the nodes in the tree and then plot the tree structure.

    Args:
        tree (Node): The root node of the decision tree to visualize.
        depth (int): The current depth of the tree, which helps in determining
                     the layout of the tree during visualization.
    """
    positions, node_ids = assign_positions(tree, depth)  # Assigns positions for each node
    plot_tree(tree, positions, node_ids, depth)  # Plots the tree using the assigned positions

