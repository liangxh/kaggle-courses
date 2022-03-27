"""
Notes

PCA 应用于已标准化 (standardized) 的数据
对已标准化的数据, variation 即 correlation
对未标准化的数据, variation 即 协方差 covariance

x' = a1 * x + b1 * y
y' = a2 * x + b2 * y
x', y' 称为主成分 principle components
[[a1, b1], [a2, b2]] 称为 loadings
"""
import commandr
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_regression
from sklearn.decomposition import PCA


class Course:
    @staticmethod
    def init_plt():
        plt.style.use("seaborn-whitegrid")
        plt.rc("figure", autolayout=True)
        plt.rc("axes", labelweight="bold", labelsize="large", titleweight="bold", titlesize=14, titlepad=10)

    @staticmethod
    def load_data():
        return pd.read_csv("../input/fe-course-data/autos.csv")

    @staticmethod
    def apply_pca(X, standardize=True):
        # 标准化
        if standardize:
            X = (X - X.mean(axis=0)) / X.std(axis=0)
        # 主成分名 PC${i}
        component_names = [f"PC{i + 1}" for i in range(X.shape[1])]

        # 训练 + 处理
        pca = PCA()
        X_pca = pca.fit_transform(X)
        X_pca = pd.DataFrame(X_pca, columns=component_names)

        # columns: 每一列 对应 一个主成分
        # index:   每一行 对应 原数据的一个特征
        loadings = pd.DataFrame(pca.components_.T, columns=component_names, index=X.columns)
        return pca, X_pca, loadings

    @staticmethod
    def plot_variance(pca, width=8, dpi=100):
        # Create figure
        fig, axs = plt.subplots(1, 2)
        n = pca.n_components_
        grid = np.arange(1, n + 1)

        # Explained variance
        evr = pca.explained_variance_ratio_
        axs[0].bar(grid, evr)
        axs[0].set(xlabel="Component", title="% Explained Variance", ylim=(0.0, 1.0))

        # Cumulative Variance
        cv = np.cumsum(evr)
        axs[1].plot(np.r_[0, grid], np.r_[0, cv], "o-")
        axs[1].set(xlabel="Component", title="% Cumulative Variance", ylim=(0.0, 1.0))

        # Set up figure
        fig.set(figwidth=width, dpi=dpi)
        return axs

    @staticmethod
    def make_mi_scores(X, y, discrete_features):
        mi_scores = mutual_info_regression(X, y, discrete_features=discrete_features)
        mi_scores = pd.Series(mi_scores, name="MI Scores", index=X.columns)
        mi_scores = mi_scores.sort_values(ascending=False)
        return mi_scores

    @staticmethod
    def examine_extreme_samples(df, X_pca, pca_name, features):
        """
        极端样本分析

        :param df: 原始数据 DataFrame
        :param X_pca: 参與 pca 的 DataFrame
        :param pca_name: 分析依据的 pca 名
        :param features: 需要显示的 df 的列名
        :return:
        """
        # 根据某个主成分的值排序的 index
        idx = X_pca[pca_name].sort_values(ascending=False).index
        print(df.loc[idx, features])

    @staticmethod
    def show_x_pca_dist(X_pca):
        sns.catplot(y="value", col="variable", data=X_pca.melt(), kind='boxen', sharey=False, col_wrap=2)

    @staticmethod
    def show_x_pca_dist_v2(X_pca):
        melt = X_pca.melt()
        melt["pca_index"] = melt["variable"].str.replace("PC", "").astype("category")
        sns.catplot(x="value", y="pca_index", data=melt, kind="boxen", height=6)


@commandr.command
def main():
    features = ["highway_mpg", "engine_size", "horsepower", "curb_weight"]
    label_y = "price"
    df = Course.load_data()

    X = df.copy()
    y = X.pop(label_y)
    X = X.loc[:, features]

    # 直接计算相关性
    print(X[features].corrwith(y))

    # pca 计算
    pca, X_pca, loadings = Course.apply_pca(X)
    # 可视化 pca 各成分
    print(loadings)
    Course.plot_variance(pca)

    # MI 分析
    mi_scores = Course.make_mi_scores(X_pca, y, discrete_features=False)
    print(mi_scores)

    # 极端样本分析
    Course.examine_extreme_samples(df, X_pca, "PC3", ["make", "body_style", "horsepower", "curb_weight"])


if __name__ == '__main__':
    commandr.Run()
