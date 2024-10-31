class Node:
    def __init__(self):
        """
        Initializes a new Node for a decision tree.

        Attributes:
            attribute (int): The index of the attribute on which to split the data.
            val (float): The value of the split (threshold) for the attribute.
            left (Node): Reference to the left child node in the tree.
            right (Node): Reference to the right child node in the tree.
            leaf (bool): A flag indicating if this node is a leaf node (i.e., it contains a class label).
        """
        self.attribute: int = None
        self.val: float = None
        self.left: 'Node' = None
        self.right: 'Node' = None
        self.leaf: bool = False

    def __str__(self) -> str:
        """
        Provides a string representation of the Node.

        Returns:
            str: A string summarizing the Node's attributes and value.
        """
        if self.leaf:
            return f"Leaf Node: Class = {self.val}"
        else:
            return f"Node: Attribute = {self.attribute}, Split Value = {self.val}"

    def __repr__(self) -> str:
        """
        Provides a formal string representation of the Node, useful for debugging.

        Returns:
            str: A string representation of the Node.
        """
        return f"Node(attribute={self.attribute}, val={self.val}, leaf={self.leaf})"

    def __hash__(self) -> int:
        """
        Returns a hash of the Node based on its attributes and values.

        Returns:
            int: Hash value of the Node.
        """
        return hash((self.attribute, self.val, self.leaf))
