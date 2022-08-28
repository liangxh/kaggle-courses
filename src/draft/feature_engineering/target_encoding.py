"""
https://www.kaggle.com/code/ryanholbrook/target-encoding/tutorial
"""
import commandr
import numpy as np
import pandas as pd
# pip install category-encoders
from category_encoders import MEstimateEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score
from xgboost import XGBRegressor


"""
Target Encoding 适用场景
- 高数量类别特征 high-cardinality features
    - one-hot encoding 的向量太长
- 领域驱动的特征 Domain-motivated features

考虑 mean encoding
autos["mean_price_by_make"] = autos.groupby("make")["price"].transform("mean")

平滑 Smoothing
- encoding = weight * in_category + (1 - weight) * overall
- M 估计, M-estimation
  - 取 weight = n / (n + m)
    - n 为对为对应类型样本数
    - m 越大, overall 的占比越大
    - 如果数据噪音较大，需要较多数据才能有准确的均值，则取 m 越大
"""


class Course:
    @staticmethod
    def load_data():
        df = pd.read_csv("../input/fe-course-data/movielens1m.csv")
        df = df.astype(np.uint8, errors='ignore')
        return df

    @staticmethod
    def split_df_for_encode(df, y_label):
        X = df.copy()

        X_encode = X.sample(frac=0.25)
        y_encode = X_encode.pop(y_label)

        # Training split
        X_pretrain = df.drop(X_encode.index)
        y_train = X_pretrain.pop(y_label)
        return (X_encode, y_encode), (X_pretrain, y_train)

    @staticmethod
    def get_m_estimate_encoder(X, y, col):
        encoder = MEstimateEncoder(cols=[col, ], m=5.0)
        # encoder.cols 可以获取 cols
        encoder.fit(X, y)
        return encoder

    @staticmethod
    def encode_m_estimate(encoder, X):
        return encoder.transform(X)

    @staticmethod
    def show_encoding_result(X, y, feature_label, y_label):
        plt.figure(dpi=90)
        # 显示 y 的不同值的出现次数分布图
        ax = sns.distplot(y, kde=False, norm_hist=True)
        # kdeplot, kernel density estimation, 核密度估计图
        ax = sns.kdeplot(X[feature_label], color='r', ax=ax)
        # 或者 sns.distplot(X[feature_label], color='r', ax=ax, kde=True, hist=False)
        # hist=True, histogram
        # kde=True, gaussian kernel density estimation 高斯核密度估计图
        # norm_hist=False, histogram 的 count 改成 density

        ax.set_xlabel(y_label)
        ax.legend(labels=[feature_label, y_label])

    @staticmethod
    def find_object_features(df):
        return df.select_dtypes(["object"]).nunique()

    @staticmethod
    def get_value_counts(df, feature):
        return df[feature].value_counts()

    @staticmethod
    def score_dataset(X, y, model=XGBRegressor()):
        for colname in X.select_dtypes(["category", "object"]):
            X[colname], _ = X[colname].factorize()
        score = cross_val_score(model, X, y, cv=5, scoring="neg_mean_squared_log_error")
        return - np.sqrt(score.mean())


@commandr.command
def main():
    feature_label = "Zipcode"
    y_label = "Rating"

    df = Course.load_data()
    (X_encode, y_encode), (X_pretrain, y_train) = Course.split_df_for_encode(df, y_label)
    encoder = Course.get_m_estimate_encoder(X_encode, y_encode, feature_label)
    X_train = Course.encode_m_estimate(encoder, X_pretrain)

    plt.figure(dpi=90)
    ax = sns.distplot(y_train, kde=False, norm_hist=True)
    ax = sns.kdeplot(X_train.Zipcode, color='r', ax=ax)
    ax.set_xlabel(y_label)
    ax.legend(labels=[feature_label, y_label])


if __name__ == '__main__':
    commandr.Run()
