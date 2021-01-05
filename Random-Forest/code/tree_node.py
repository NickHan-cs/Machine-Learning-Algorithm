class TreeNode:
    def __init__(self):
        self.left_node = None
        self.right_node = None
        self.type = None    # "internal" / "leaf"
        # 当type为"internal"时，split_feature和split_feature_value为划分的特征和划分的特征取值
        self.split_feature = None
        self.split_feature_value = None
        # 当type为"leaf"时，category为该叶子结点所属的label
        self.category = None

    def predict(self, x):
        tree_node = self
        while tree_node.type != "leaf":
            if tree_node.split_feature_value == x[tree_node.split_feature]:
                tree_node = tree_node.left_node
            else:
                tree_node = tree_node.right_node
        return tree_node.category
