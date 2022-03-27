import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor


def train_decision_tree_regressor(X, y):
    train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=0)

    max_leaf_nodes_candidates = [10, 20, 50, 100, 200]
    max_leaf_nodes_mae = dict()
    for max_leaf_nodes in max_leaf_nodes_candidates:
        model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
        model.fit(train_X, train_y)
        preds_val = model.predict(val_X)
        mae = mean_absolute_error(val_y, preds_val)
        max_leaf_nodes_mae[max_leaf_nodes] = mae

    best_max_leaf_nodes = sorted(max_leaf_nodes_mae.items(), key=lambda _item: _item[1])[0][0]
    model = DecisionTreeRegressor(max_leaf_nodes=best_max_leaf_nodes, random_state=0)
    model.fit(X, y)
    predicted = model.predict(X)
    mae = mean_absolute_error(y, predicted)
    return mae


def train_random_forest(X, y):
    model = RandomForestRegressor(random_state=0)
    model.fit(X, y)
    predicted = model.predict(X)
    mae = mean_absolute_error(y, predicted)
    return mae


def main():
    melbourne_file_path = '../../input/melbourne-housing-snapshot/melb_data.csv'
    melbourne_data = pd.read_csv(melbourne_file_path)
    melbourne_data.describe()

    melbourne_data = melbourne_data.dropna(axis=0)

    y = melbourne_data.Price
    melbourne_features = ['Rooms', 'Bathroom', 'Landsize', 'BuildingArea', 'YearBuilt', 'Lattitude', 'Longtitude']
    X = melbourne_data[melbourne_features]

    mae = train_decision_tree_regressor(X, y)
    print("decision tree: {}".format(mae))

    mae = train_random_forest(X, y)
    print("random forest: {}".format(mae))


if __name__ == '__main__':
    main()
