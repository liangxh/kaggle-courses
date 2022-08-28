import numpy as np
import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_regression


class MiScore:
    """
    Mutual Information Score
    """
    def __init__(self, X: DataFrame, y: DataFrame):
        X = X.copy()

        # 将所有离散值转换成 id
        for col in X.select_dtypes(["object", "category"]):
            X[col], _ = X[col].factorize()

        discrete_features = [pd.api.types.is_integer_dtype(t) for t in X.dtypes]
        mi_scores = mutual_info_regression(X, y, discrete_features=discrete_features, random_state=0)

        # 转换成 pd.DataFrame
        mi_scores = pd.Series(mi_scores, name="MI Scores", index=X.columns)

        # 按 mi score 降序顺序
        mi_scores = mi_scores.sort_values(ascending=False)
        self.mi_scores = mi_scores

    def get(self):
        return self.mi_scores

    def drop_uninformative(self, df):
        return df.loc[:, self.mi_scores > 0.0]

    def plot(self):
        scores = self.mi_scores.sort_values(ascending=True)
        width = np.arange(len(scores))
        ticks = list(scores.index)
        plt.barh(width, scores)
        plt.yticks(width, ticks)
        plt.title("Mutual Information Scores")
