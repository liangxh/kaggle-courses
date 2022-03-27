import commandr
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.feature_selection import mutual_info_regression
# from sklearn.feature_selection import mutual_info_classif
import matplotlib.pyplot as plt


class Course:
    @staticmethod
    def init_plt():
        # plot 设置初始化
        plt.style.use("seaborn-whitegrid")
        plt.rc("figure", autolayout=True)
        plt.rc("axes", labelweight="bold", labelsize="large", titleweight="bold", titlesize=14, titlepad=10)

    @staticmethod
    def load_data():
        # 数据读取
        df = pd.read_csv("../../input/fe-course-data/ames.csv")
        X = df.copy()
        y = X.pop("SalePrice")
        return df, X, y

    @staticmethod
    def factorize_object_and_category(X):
        for col_name in X.select_dtypes("object", "category"):
            X[col_name], _ = X[col_name].factorize()
        return X

    @staticmethod
    def get_discrete_features(X):
        return [pd.api.types.is_integer_dtype(t) for t in X.dtypes]

    @staticmethod
    def show_plots(df):
        # 显示两个字段的分布
        sns.relplot(x="curb_weight", y="price", data=df)

        # 不同的 hue 对应数据会显示为不同颜色
        sns.lmplot(x="horsepower", y="price", hue="fuel_type", data=df)

        # df.melt
        # 保留 id_vars 字段
        # value_vars 对应多个特征，会转换成 variable 和 value 两个列
        # 对 df.count() = m, len(value_vars) = n, 将生成 m*n 行，3 列的表
        features = ["YearBuilt", "MoSold", "ScreenPorch"]
        # col 字段: 按不到的 variable 值分成不同的子图
        sns.relplot(
            x="value", y="SalePrice",
            col="variable", data=df.melt(id_vars="SalePrice", value_vars=features),
            facet_kws=dict(sharex=False),
        )

    """
    Mutual Information 计算
    """
    @staticmethod
    def calculate_mi_scores(X, y, discrete_features):
        mi_scores = mutual_info_regression(X, y, discrete_features=discrete_features)
        mi_scores = pd.Series(mi_scores, name="MI Scores", index=X.columns)
        mi_scores = mi_scores.sort_values(ascending=False)
        return mi_scores

    @staticmethod
    def plot_mi_scores(scores):
        plt.figure(dpi=100, figsize=(8, 5))
        scores = scores.sort_values(ascending=True)
        width = np.arange(len(scores))
        ticks = list(scores.index)
        plt.barh(width, scores)
        plt.yticks(width, ticks)
        plt.title("Mutual Information Scores")


@commandr.command
def test_plots():
    Course.init_plt()
    df, _, _ = Course.load_data()
    Course.show_plots(df)


@commandr.command
def test_mi_scores():
    Course.init_plt()
    df, X, y = Course.load_data()
    X = Course.factorize_object_and_category(X)
    discrete_features = Course.get_discrete_features(X)
    mi_scores = Course.calculate_mi_scores(X, y, discrete_features)

    # print(mi_scores.head(20))
    # print(mi_scores.tail(20))

    Course.plot_mi_scores(mi_scores.head(20))


if __name__ == "__main__":
    commandr.Run()
