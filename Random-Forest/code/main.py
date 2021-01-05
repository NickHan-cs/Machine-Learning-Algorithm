import json
import pandas as pd
from random_forest import RandomForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

if __name__ == "__main__":
    with open("config.json", "r") as config_file:
        config = json.load(config_file)
    if config["is_train"]:
        x_train = pd.read_csv("../data/train/x_train.csv")
        y_train = pd.read_csv("../data/train/y_train.csv")
        train_dataset = pd.merge(x_train, y_train)
        train_data, eval_data = train_test_split(train_dataset, test_size=0.2)
        random_forest = RandomForest(max_depth=config["max_depth"],
                                     min_samples_split=config["min_samples_split"],
                                     min_impurity_split=config["min_impurity_split"],
                                     max_features=config["max_features"],
                                     n_estimators=config["n_estimators"])
        random_forest.fit(train_data)
        random_forest.save_model()
        pred_label = random_forest.predict(eval_data)
        print("ac: ", accuracy_score(eval_data['label'], pred_label))
        print("precision: ", precision_score(eval_data['label'], pred_label))
        print("recall: ", recall_score(eval_data['label'], pred_label))
        print("f1_score: ", f1_score(eval_data['label'], pred_label))
    else:
        x_data = pd.read_csv("../data/train/x_train.csv")
        y_data = pd.read_csv("../data/train/y_train.csv")
        test_data = pd.merge(x_data, y_data)
        model_file = open(config["model_file"], 'r')
        lines = model_file.readlines()
        params = lines[0].split(' ')
        max_depth = int(params[0]) if params[0] is not None else None
        min_samples_split = int(params[1])
        min_impurity_split = float(params[2])
        max_features = params[3] if params[3] == "None" or params[3] == "log2" or params[3] == "sqrt" else float(
            params[3])
        n_estimators = int(params[4])
        random_forest = RandomForest(max_depth=max_depth, min_samples_split=min_samples_split,
                                     min_impurity_split=min_impurity_split,
                                     max_features=max_features, n_estimators=n_estimators)
        random_forest.load_model(config["model_file"])
        pred_label = random_forest.predict(test_data)
        print("ac: ", accuracy_score(test_data['label'], pred_label))
        print("precision: ", precision_score(test_data['label'], pred_label))
        print("recall: ", recall_score(test_data['label'], pred_label))
        print("f1_score: ", f1_score(test_data['label'], pred_label))
