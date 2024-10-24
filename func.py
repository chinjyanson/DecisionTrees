import numpy as np
from node_class import Node

def parse(dataset_file_name):
    """
    Parse the txt file into wifi components
    """
    line = np.loadtxt(f"wifi_db/{dataset_file_name}.txt", delimiter='\t')
    return line


def decision_tree_learning(train: list[list[int]], depth: int) -> tuple:
# arguments: Matrix containing data set and depth variable
    """
    procedure decision tree learning(training dataset, depth)
        if all samples have the same label then
            return (a leaf node with this value, depth)
        else
            split← find split(training dataset)
            node← a new decision tree with root as split value
            l branch, l depth ← DECISION TREE LEARNING(l dataset, depth+1)
            r branch, r depth ← DECISION TREE LEARNING(r dataset, depth+1)
            return (node, max(l depth, r depth))
        end if  
    end procedure
    """

    class_labels = [row[-1] for row in train]

    # Base case: if all samples have the same label, return a leaf node
    if ((np.size(np.unique(class_labels)) == 1) | (depth >= 5)):
        return (class_labels[0], depth)
    
    else:
        split = find_split(train)  # Find the best attribute and value to split on
    
        # node = {'attribute': split["attribute"], 'value': split["value"], 'left': None, 'right': None}
        node = Node()
        node.attribute = split["attribute"]
        node.val = split["value"]

        # left_table = train[:split["attribute"]]

        left_table = [row for row in train if row[split["attribute"]] <= split["value"]]
        right_table = [row for row in train if row[split["attribute"]] > split["value"]]

        # Check if empty cos otherwise doesnt work but do we want to do that?
        if len(left_table) == 0 or len(right_table) == 0:
            return (class_labels[0], depth)
    
        left_branch, left_depth = decision_tree_learning(left_table, depth + 1)
        right_branch, right_depth = decision_tree_learning(right_table, depth + 1)
    
        node.left = left_branch
        node.right = right_branch

        return (node, max(left_depth, right_depth))

def find_entropy(dataset):
    """
    This function finds the entropy of the dataset
    """
    labels = []
    for line in dataset:
        labels.append(line[-1])
    
    unique, counts = np.unique(labels, return_counts=True)
    percentages = counts/counts.sum()

    entropy_array = -percentages*np.log2(percentages)
    entropy = entropy_array.sum()
    count = len(labels)
    return entropy, count

def find_split(dataset: list[list[int]]):    #calculate Information Gain
    """
    This function finds the most optimal/highest information gain
    """
    best_split = {"attribute": 100, "value": 100, "entropy": 100}

    #for loop that sorts wifi1-wifi7, in decreasing value for each wifi (just wifi and room)
    number_attributes = len(dataset[0]) -1
    for k in range (number_attributes):
        wifi_table = [[row[k], row[-1]] for row in dataset] #takes wifi column and class column
        wifi_table = sorted(wifi_table, key=lambda x: x[0]) #sorts it in ascending order

    #after sorting, we identify every room change and identify a cut value 
        for i in range(len(wifi_table) -1 ):
            if (wifi_table[i][1] != wifi_table[i+1][1]):
                split_value = wifi_table[i][0]

    #we also calculate the weighted average entropy of the produced subsets of each cut
                entropy_left, count_left = find_entropy(wifi_table[:i])
                entropy_right, count_right = find_entropy(wifi_table[i:])
                total_count = count_left + count_right
                weighted_entropy = (count_left/total_count)*entropy_left + (count_right/total_count)*entropy_right

    #then we compare this weighted average entropy to the current minimum value we have. If smaller, store in best_cut tuple = <attribute, value, entropy>
                if weighted_entropy < best_split["entropy"]:
                    best_split["entropy"] = weighted_entropy
                    best_split["value"] = split_value
                    best_split["attribute"] = k+1

    #we exit the for loop and return this best_cut tuple
    return best_split

def predict():
    pass


if __name__ == "__main__":
    data = parse("clean_dataset")
    print(decision_tree_learning(data, 1))