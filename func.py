import numpy as np
from node_class import Node
import evaluation as eval

def parse(dataset_file_name):
    """
    Parse the txt file into wifi components
    """
    x = []
    y = []
    with open(f"wifi_db/{dataset_file_name}.txt", 'r') as f:
        for line in f:
            if line.strip() != "":
                row = line.strip().split()
                x.append(list(map(float,row[:-1])))
                y.append(float(row[-1]))
    x = np.array(x)
    y = np.array(y)
    
    return x, y                                                                                                                                                                                                                                                                       

def decision_tree_learning(train: list[list[float]], depth: int) -> tuple:
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

    #Check if we have reached the maximum depth
    #If so, take the majority label of the remaining datapoints
    if (depth >= 8):
        leaf_node = Node()
        leaf_node.leaf = True
        unique_classes, counts = np.unique(class_labels, return_counts=True)
        leaf_node.val = int(unique_classes[np.argmax(counts)])
        leaf_node.attribute = "Leaf"
        print("hit depth")
        return (leaf_node, depth)

    #Check if all samples have the same label, return a leaf node if so 
    elif (len(np.unique(class_labels)) == 1):
        leaf_node = Node()
        leaf_node.leaf = True
        leaf_node.val = int(np.unique(class_labels))
        print(class_labels)
        leaf_node.attribute = "Leaf"
        print("same")
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
            print("special case")
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

       

def find_entropy(dataset):
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


def find_split(dataset: list[list[float]]):  #calculate Information Gain
    """
    This function finds the most optimal/highest information gain
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
                entropy_left, count_left = find_entropy(wifi_table[:i+1])
                entropy_right, count_right = find_entropy(wifi_table[i+1:])
                total_count = count_left + count_right
                weighted_entropy = (count_left/total_count)*entropy_left + (count_right/total_count)*entropy_right

    #then we compare this weighted average entropy to the current minimum value we have. If smaller, store in best_cut map = <attribute, value, entropy>
                if (weighted_entropy < best_split["entropy"]):
                    best_split["entropy"] = weighted_entropy
                    best_split["value"] = split_value
                    best_split["attribute"] = k

    #we exit the for loop and return this best_cut tuple
    return best_split

if __name__ == "__main__":
    data = parse("clean_dataset")
    x_train, x_test, y_train, y_test = eval.split_dataset(data[:, :-1], data[:, -1], 0.2)
    dataset = list(zip(x_train, y_train))
