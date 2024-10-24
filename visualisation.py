"""
Used to generate and visualise the decision tree and the data splot
"""

import matplotlib.pyplot as plt
import numpy as np

# Function to assign positions to the tree nodes
def assign_positions(tree, x=0, y=0, vertical_dist=1, horizontal_dist=2, positions=None):
    if positions is None:
        positions = {}

    # Assign current node position
    positions[tree['value']] = (x, y, tree['attribute'])

    # Calculate positions for the left and right children
    if tree.get('left'):
        positions = assign_positions(tree['left'], x - horizontal_dist, y - vertical_dist, vertical_dist, horizontal_dist / 1.5, positions)
    if tree.get('right'):
        positions = assign_positions(tree['right'], x + horizontal_dist, y - vertical_dist, vertical_dist, horizontal_dist / 1.5, positions)

    return positions

# Function to plot the tree using the calculated positions
def plot_tree(tree, positions):
    fig, ax = plt.subplots()

    # Plot the edges
    def plot_edges(tree, positions):
        if tree.get('left'):
            x_values = [positions[tree['value']][0], positions[tree['left']['value']][0]]
            y_values = [positions[tree['value']][1], positions[tree['left']['value']][1]]
            ax.plot(x_values, y_values, 'k-', lw=1)  # black line for edges
            plot_edges(tree['left'], positions)
        if tree.get('right'):
            x_values = [positions[tree['value']][0], positions[tree['right']['value']][0]]
            y_values = [positions[tree['value']][1], positions[tree['right']['value']][1]]
            ax.plot(x_values, y_values, 'k-', lw=1)
            plot_edges(tree['right'], positions)

    plot_edges(tree, positions)

    # Plot the nodes
    for node, (x, y, attribute) in positions.items():
        ax.text(x, y, str(f"{attribute} < {node}"), fontsize=12, ha='center', va='center', bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', edgecolor='black'))

    ax.set_aspect('equal')
    ax.set_axis_off()
    plt.savefig("tree.png")

binary_tree = {
    'value': 1,
    'attribute': 'X1',
    'left': {
        'value': 2,
        'attribute': 'X2',
        'left': {'value': 4, 'attribute': 'X5'},
        'right': {'value': 5, 'attribute': 'X3'}
    },
    'right': {
        'value': 3,
        'attribute': 'X4',
        'right': {'value': 6, 'attribute': 'X6'}
    }
}

# Assign positions to the tree nodes
positions = assign_positions(binary_tree)

# Plot the tree
plot_tree(binary_tree, positions)
