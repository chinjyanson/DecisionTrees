import numpy as np
from node_class import Node
from typing import Dict, List, Tuple

def parse(dataset_file_name: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Parses a dataset file of WiFi signal strengths and labels.

    Args:
        dataset_file_name (str): The name of the dataset file (without extension),
                                 located in the "wifi_db/" directory.

    Returns:
        Tuple[numpy.ndarray, numpy.ndarray]:
            A tuple containing:
            - x (numpy.ndarray): Array of WiFi signal strengths, with shape (n_samples, n_features).
            - y (numpy.ndarray): Array of corresponding labels, with shape (n_samples,).
    """
    x, y = [], []

    # Read the dataset file and parse contents
    with open(f"wifi_db/{dataset_file_name}.txt", 'r') as file:
        for line in file:
            line = line.strip()
            if line:
                row = list(map(float, line.split()))
                x.append(row[:-1])  # Signal strengths (features)
                y.append(row[-1])   # Label

    # Convert lists to numpy arrays
    x = np.array(x)
    y = np.array(y)

    return x, y

def decision_tree_learning(train: list[list[float]], depth: int) -> tuple:
    """
    This function builds the decision tree recursively.
    Params: train: list[list[float]]: The training dataset
            depth: int: The current depth of the tree
    """

    class_labels = [row[-1] for row in train]

    #Check if all samples have the same label, return a leaf node if so
    if (len(np.unique(class_labels)) == 1):
        leaf_node = Node()
        leaf_node.leaf = True
        leaf_node.val = int(np.unique(class_labels))
        leaf_node.attribute = "Leaf"
        return (leaf_node, depth)

    else:
        split = find_split(train)  # Find the best attribute and value to split on

        #However if our dataset has only one possible split, and the 2 datapoints of this split have the same attribute value but different class:
        #This split should not be considered and hence no entropy will be calculated. The "best entropy" value would hence never be updated from 100 (arbitrary value).
        #We should return a leaf node with the majority label of the remaining datapoints
        if (split["entropy"] == 100):
            leaf_node = Node()
            leaf_node.leaf = True
            unique_classes, counts = np.unique(class_labels, return_counts=True)
            leaf_node.val = int(unique_classes[np.argmax(counts)])
            leaf_node.attribute = "Leaf"
            return (leaf_node, depth)
        else:
            node = Node()
            node.attribute = split["attribute"] + 1
            node.val = split["value"]

            left_table = [row for row in train if row[split["attribute"]] <= split["value"]]
            right_table = [row for row in train if row[split["attribute"]] > split["value"]]

            left_branch, left_depth = decision_tree_learning(left_table, depth + 1)
            node.left = left_branch

            right_branch, right_depth = decision_tree_learning(right_table, depth + 1)
            node.right = right_branch
            
            return (node, max(left_depth, right_depth))

def find_entropy(dataset: list[list[float]]) -> tuple:
    """
    Calculates the entropy of the given dataset.
    """
    labels = [row[-1] for row in dataset]
    
    if len(labels) == 0:
        return 0, 0
    
    _, counts = np.unique(labels, return_counts=True)
    percentages = counts / counts.sum()

    entropy_array = -percentages * np.log2(percentages)
    entropy = entropy_array.sum()
    count = len(labels)
    return entropy, count

def find_split(dataset: list[list[float]]) -> dict:
    """
    This function finds the most optimal/highest information gain
    Params: dataset: list[list[float]]: The dataset to find the split on
    """
    best_split = {"attribute": 100, "value": 100, "entropy": 100}

    #for loop that sorts wifi1-wifi7, in decreasing value for each wifi (just wifi and room)
    number_attributes = len(dataset[0])-1
    for k in range (0, number_attributes):
        wifi_table = [[row[k], row[-1]] for row in dataset] #takes wifi column and class column
        wifi_table = sorted(wifi_table, key=lambda x: x[0]) #sorts it in ascending order

    #after sorting, we identify every room change and identify a cut value
    #iterate only len(wifi_table)-1 times because we dont evaluate the last line (can't have a split with 0 entries)
        for i in range(len(wifi_table) -1):
            if ((wifi_table[i][1] != wifi_table[i+1][1]) & (wifi_table[i][0] != wifi_table[i+1][0])):
                split_value = wifi_table[i][0]

    #we also calculate the weighted average entropy of the produced subsets of each cut
                left_entropy, left_count = find_entropy(wifi_table[:i+1])
                right_entropy, right_count = find_entropy(wifi_table[i+1:])
                total_count = left_count + right_count
                weighted_entropy = (left_count/total_count)*left_entropy + (right_count/total_count)*right_entropy

    #then we compare this weighted average entropy to the current minimum value we have. If smaller, store in best_cut map = <attribute, value, entropy>
                if (weighted_entropy < best_split["entropy"]):
                    best_split["entropy"] = weighted_entropy
                    best_split["value"] = split_value
                    best_split["attribute"] = k

    #we exit the for loop and return this best_cut tuple
    return best_split

def find_split2(dataset: List[List[float]]) -> Dict[str, float]:
    """
    Finds the optimal attribute and value to split the dataset based on information gain.

    Args:
        dataset (List[List[float]]): The dataset to evaluate for potential splits,
                                      where each inner list represents a sample 
                                      with features and the last element as the label.

    Returns:
        Dict[str, float]:
            A dictionary containing the best split information:
            - "attribute": Index of the attribute used for the split.
            - "value": The value of the attribute at which to split.
            - "entropy": The entropy of the resulting split.
    """
    best_split = {"attribute": -1, "value": None, "entropy": float('inf')}
    number_of_attributes = len(dataset[0]) - 1

    # Iterate over each attribute to find the optimal split
    for attribute_index in range(number_of_attributes):
        # Create a sorted list of attribute values with corresponding labels
        sorted_wifi_table = sorted(
            [(row[attribute_index], row[-1]) for row in dataset],
            key=lambda x: x[0]
        )

        # Iterate through sorted values to find potential split points
        for i in range(len(sorted_wifi_table) - 1):
            # Check for a change in class label to find a split point
            if sorted_wifi_table[i][1] != sorted_wifi_table[i + 1][1]:
                split_value = sorted_wifi_table[i][0]

                # Calculate the weighted average entropy of the resulting subsets
                left_entropy, left_count = find_entropy(sorted_wifi_table[:i + 1])
                right_entropy, right_count = find_entropy(sorted_wifi_table[i + 1:])
                total_count = left_count + right_count
                weighted_entropy = (
                    (left_count / total_count) * left_entropy +
                    (right_count / total_count) * right_entropy
                )

                # Update the best split if the new weighted entropy is lower
                if weighted_entropy < best_split["entropy"]:
                    best_split.update({"entropy": weighted_entropy, "value": split_value, "attribute": attribute_index})

    return best_split

