class DecisionTree():
    def __init__(self):
        self.val = 
        self.left = 
        self.right = 
        self.leaf = False


    # def fit(self, X, y):
    #     pass

    # def predict(self, X):
    #     pass

    def decision_tree_learning(matrix, depth):
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
        pass

    def find_split():
        """
        This function finds the most optimal/highest information gain
        """
        pass


if __name__ == "__main__":
    dtree = DecisionTree()
    dtree.decision_tree_learning()
    # then visualise it