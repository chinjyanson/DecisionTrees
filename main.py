import numpy as np
#import matplotlib.pyplot as plt
import utils


class DecisionTree():
    def __init__(self):
        self.attribute = None 
        self.val = None
        self.left = None
        self.right = None
        self.leaf = False

    # def decision_tree_learning(self, train: list[list[int]], depth: int) -> tuple:
    #     # arguments: Matrix containing data set and depth variable

    #     """
    #     procedure decision tree learning(training dataset, depth)
    #         if all samples have the same label then
    #             return (a leaf node with this value, depth)
    #         else
    #             split← find split(training dataset)
    #             node← a new decision tree with root as split value
    #             l branch, l depth ← DECISION TREE LEARNING(l dataset, depth+1)
    #             r branch, r depth ← DECISION TREE LEARNING(r dataset, depth+1)
    #             return (node, max(l depth, r depth))
    #         end if  
    #     end procedure
    #     """
    #     return ()

    def find_split(self, dataset: list[list[int]]) -> tuple:    #calculate Information Gain
        """
        This function finds the most optimal/highest information gain
        """
        pass

    def find_entropy(self, dataset):
        """
        This function finds the entropy of the dataset
        """
        labels = []
        for line in dataset:
            labels.append(line[7])
        #print("labels :")
        #print(labels)
        unique, counts = np.unique(labels, return_counts=True)
        # finds percentage of labels throughout given dataset
        #calculates entropy according to formula
        pass

    def plot_tree(self, ax, x, y, dx, dy):
        """
        Recursively plot the decision tree using matplotlib.
        """
        if self.leaf:
            ax.text(x, y, f"Leaf: {self.val}", ha='center', bbox=dict(facecolor='lightgreen', edgecolor='black'))
        else:
            ax.text(x, y, f"X{self.attribute} <= {self.val}", ha='center', bbox=dict(facecolor='lightblue', edgecolor='black'))
            
            # Recursively plot the left and right branches
            if self.left:
                ax.plot([x, x - dx], [y, y - dy], 'k-')  # Draw line to the left child
                self.left.plot_tree(ax, x - dx, y - dy, dx / 2, dy)  # Plot the left child
            if self.right:
                ax.plot([x, x + dx], [y, y - dy], 'k-')  # Draw line to the right child
                self.right.plot_tree(ax, x + dx, y - dy, dx / 2, dy)  # Plot the right child

    def predict():
        pass


# if __name__ == "__main__":
#     dtree = DecisionTree()
#     dtree.decision_tree_learning()
#     # then visualise it

tree_classifier = DecisionTree()
dataset = utils.start()
tree_classifier.find_entropy(dataset)