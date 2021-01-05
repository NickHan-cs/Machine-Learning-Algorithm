import numpy as np
import pandas as pd
from tree_node import TreeNode


def compute_gini(y):
    gini = 1
    for cate in y.value_counts(1):
        gini -= cate * cate
    return gini


class DecisionTreeCART:
    def __init__(self, max_depth=None, min_samples_split=2, min_impurity_split=1e-7, max_features=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_impurity_split = min_impurity_split
        self.max_features = max_features
        self.root_node = TreeNode()

    def _tree_generate(self, tree_node, train_data, depth):
        if train_data.shape[0] < self.min_samples_split \
                or train_data.shape[1] <= 2 \
                or self.max_depth is not None and depth == self.max_depth \
                or compute_gini(train_data['label']) < self.min_impurity_split:
            '''
            不再继续分裂的条件：
            当结点的样本数少于min_sample_split时
            当样本没有更多特征时
            当树的深度达到决策树的max_depth时
            当结点的样本集的Gini系数小于min_impurity_split时
            '''
            tree_node.type = 'leaf'
            tree_node.category = train_data['label'].value_counts(ascending=False).keys()[0]
            return
        # feature2values记录feature及其可能的取值
        feature2values = dict()
        for i in range(len(train_data.columns)):
            feature = train_data.columns[i]
            if feature != 'index' and feature != 'label' and pd.unique(train_data[feature]).shape[0] > 1:
                feature2values[feature] = list(train_data[feature].value_counts().keys())
        # 根据max_features选取特征
        chosen_features = list(feature2values.keys())
        chosen_features_num = len(chosen_features)
        if self.max_features == "log2":
            chosen_features_num = np.log2(len(chosen_features))
        elif self.max_features == "sqrt":
            chosen_features_num = np.sqrt(len(chosen_features))
        elif 0 < self.max_features < 1:
            chosen_features_num = len(chosen_features) * self.max_features
        chosen_features = np.random.choice(chosen_features, size=np.ceil(chosen_features_num).astype(int))
        # 枚举选取最优划分特征和特征值
        min_gini_index = 1
        split_feature = None
        split_feature_value = 0
        for feature in chosen_features:
            for value in feature2values[feature]:
                dv = train_data[train_data[feature] == value]
                dv_nums = dv.shape[0]
                gini_dv = compute_gini(dv['label'])
                dv_not = train_data[train_data[feature] != value]
                dv_not_nums = dv_not.shape[0]
                gini_dv_not = compute_gini(dv_not['label'])
                gini_index = (dv_nums * gini_dv + dv_not_nums * gini_dv_not) / (dv_nums + dv_not_nums)
                if gini_index < min_gini_index:
                    min_gini_index = gini_index
                    split_feature = feature
                    split_feature_value = value
        # 划分样本集
        left_data = train_data[train_data[split_feature] == split_feature_value].drop([split_feature], axis=1)
        right_data = train_data[train_data[split_feature] != split_feature_value]
        if pd.unique(right_data[split_feature]).shape[0] <= 1:
            right_data = right_data.drop([split_feature], axis=1)
        # 设置当前结点的属性
        tree_node.type = "internal"
        tree_node.split_feature = split_feature
        tree_node.split_feature_value = split_feature_value
        tree_node.left_node = TreeNode()
        tree_node.right_node = TreeNode()
        # 递归构建左右子结点
        self._tree_generate(tree_node=tree_node.left_node, train_data=left_data, depth=depth+1)
        self._tree_generate(tree_node=tree_node.right_node, train_data=right_data, depth=depth+1)

    def fit(self, train_data):
        self._tree_generate(tree_node=self.root_node, train_data=train_data, depth=0)

    def predict(self, x_data):
        y_data = np.zeros((x_data.shape[0], 1))
        for i in range(x_data.shape[0]):
            y_data[i] = self.root_node.predict(x_data.iloc[i])
        return y_data
