import numpy as np
import pandas as pd
from tree_node import TreeNode
from decision_tree import DecisionTreeCART
import time


class RandomForest:
    def __init__(self, max_depth=None, min_samples_split=2, min_impurity_split=1e-7, max_features=None,
                 n_estimators=10):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_impurity_split = min_impurity_split
        self.max_features = max_features
        self.n_estimators = n_estimators
        self.decision_trees = []

    def fit(self, train_data):
        drop_columns = []
        for i in range(len(train_data.columns)):
            feature = train_data.columns[i]
            if feature != 'index' and feature != 'label' and pd.unique(train_data[feature]).shape[0] == 1:
                drop_columns.append(feature)
        train_data = train_data.drop(drop_columns, axis=1)
        for i in range(self.n_estimators):
            decision_tree = DecisionTreeCART(max_depth=self.max_depth, min_samples_split=self.min_samples_split,
                                             min_impurity_split=self.min_impurity_split,
                                             max_features=self.max_features)
            decision_tree.fit(train_data)
            print(f"tree {i} done")
            self.decision_trees.append(decision_tree)

    def predict(self, x_data):
        y_data = np.zeros((x_data.shape[0], 1))
        print("prediction start...")
        for i in range(x_data.shape[0]):
            if i % 1000 == 0:
                print(f"{i} samples done")
            pred_dict = dict()
            for j in range(len(self.decision_trees)):
                pred_rlt = self.decision_trees[j].root_node.predict(x_data.iloc[i])
                if pred_rlt not in pred_dict:
                    pred_dict[pred_rlt] = 0
                pred_dict[pred_rlt] += 1
            y_data[i] = max(pred_dict, key=pred_dict.get)
        print(f"{x_data.shape[0]} samples all done")
        print("prediction finish...")
        return y_data

    def save_model(self):
        now = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime(time.time()))
        model_file = open(f'../model/{now}_rf_model.txt', 'w')
        model_file.write("{} {} {} {} {}\n".format(self.max_depth if self.max_depth is not None else "None",
                                                   self.min_samples_split,
                                                   self.min_impurity_split,
                                                   self.max_features if self.max_features is not None else "None",
                                                   self.n_estimators))
        for i in range(len(self.decision_trees)):
            model_file.write(f"tree {i}:\n")
            node_list = [self.decision_trees[i].root_node]
            node2num = {node_list[0]: 0}
            num = 1
            while len(node_list) != 0:
                cur_node = node_list.pop(0)
                if cur_node.left_node is not None:
                    node2num[cur_node.left_node] = num
                    num += 1
                    node_list.append(cur_node.left_node)
                if cur_node.right_node is not None:
                    node2num[cur_node.right_node] = num
                    num += 1
                    node_list.append(cur_node.right_node)
                model_file.write('{} {} {} {} {} {} {}\n'.format(node2num[cur_node], cur_node.split_feature,
                                                                 cur_node.split_feature_value, cur_node.type,
                                                                 node2num[
                                                                     cur_node.left_node] if cur_node.left_node is not None else "None",
                                                                 node2num[
                                                                     cur_node.right_node] if cur_node.right_node is not None else "None",
                                                                 cur_node.category))
            model_file.write(f"end tree {i}\n")
        model_file.close()

    def load_model(self, model_file_name):
        model_file = open(f"../model/{model_file_name}", 'r')
        lines = model_file.readlines()
        for i in range(self.n_estimators):
            self.decision_trees.append(DecisionTreeCART(max_depth=self.max_depth,
                                                        min_samples_split=self.min_samples_split,
                                                        min_impurity_split=self.min_impurity_split,
                                                        max_features=self.max_features))
        line_no = 1
        tree_cnt = 0
        for tree in self.decision_trees:
            num2node = {0: tree.root_node}
            if lines[line_no][0] == "t":
                line_no += 1
            while line_no < len(lines) and lines[line_no][0] != "e":
                node_info = lines[line_no].replace("\n", '').split(' ')
                num = int(node_info[0])
                cur_node = num2node[num]
                cur_node.split_feature = node_info[1] if node_info[1] != "None" else None
                cur_node.split_feature_value = int(node_info[2]) if node_info[2] != "None" else None
                cur_node.type = node_info[3]
                if node_info[4] != "None":
                    left_node = TreeNode()
                    cur_node.left_node = left_node
                    num2node[int(node_info[4])] = left_node
                if node_info[5] != "None":
                    right_node = TreeNode()
                    cur_node.right_node = right_node
                    num2node[int(node_info[5])] = right_node
                if node_info[6] != "None":
                    cur_node.category = int(node_info[6])
                line_no += 1
            print(f"building tree {tree_cnt} finish...\n")
            tree_cnt += 1
            line_no += 1
