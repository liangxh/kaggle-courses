import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


class CreateFeature:
    @staticmethod
    def func_log1p(df, name):
        # np.log1p(x) = log(1+x)
        return df[name].apply(np.log1p)

    @staticmethod
    def show_tables(df):
        # 作图
        fig, axs = plt.subplots(1, 2, figsize=(8, 4))
        sns.kdeplot(df["WindSpeed"], shade=True, ax=axs[0])
        sns.kdeplot(df["LogWindSpeed"], shade=True, ax=axs[1])

    @staticmethod
    def count_tree_features(df, features):
        # 计算 true 的特征数量
        return df[features].sum(axis=1)

    @staticmethod
    def count_positive_features(df, features):
        # 计算 > 0 值的特征数量
        return df[features].gt(0).sum(axis=1)

    @staticmethod
    def split_column_into_two(df, src_name, dst_names, n=-1):
        # 字段串分割
        # 此处若 n 为默认值, 则分割所有列, 序号 0 ~ N-1
        # 若 n >= 1, 则最多分割出 n+1 列, 序号 0 ~ n
        df[dst_names] = df[src_name].str.split(" ", n=n, expand=True)
        return df

    @staticmethod
    def concat_columns(df, src_names, dst_name, separator="_"):
        # 合并字符串
        dst = df[src_names[0]]
        for name in src_names[1:]:
            dst += separator + df[name]
        df[dst_name] = dst

    # Group Transforms
    @staticmethod
    def group_by_and_mean(df, group_by_name,  src_name):
        return df.groupby(group_by_name)[src_name].transform("mean")

    @staticmethod
    def group_by_and_count(df, group_by_name,  src_name):
        return df.groupby(group_by_name)[src_name].transform("count")

    @staticmethod
    def get_group_by_distribution(df, group_by_name,  src_name):
        # 注意分母用的是 df[src_name].count() 而不是 df.count()
        # 因为 df.count() 会返回一个长度 == 列数的 Series
        # 取任意一个 feature 只后再 count 则返回单个值, 但其实 len(df) 应该也可以
        return df.groupby(group_by_name)[src_name].transform("count") / df[src_name].count()

    @staticmethod
    def merge_group_by_features(df):
        # Create splits
        df_train = df.sample(frac=0.5)
        df_valid = df.drop(df_train.index)

        # 计算不同 Coverage 下 ClaimAmount 的均值
        # 注意这里 df_train["AverageClaim"].count() == df.count(), 因此下面要做 drop_duplicates()
        df_train["AverageClaim"] = df_train.groupby("Coverage")["ClaimAmount"].transform("mean")

        # 透过 merge 为 valid 添加和 train 相同的值
        # merge 相当于 join
        # 但 pd 中 join 是两个行数相同的 序列直接拼在一起
        df_valid = df_valid.merge(df_train[["Coverage", "AverageClaim"]].drop_duplicates(), on="Coverage", how="left")
        return df_train, df_valid

    @staticmethod
    def to_one_hot(df, names, prefix="cat"):
        # names 是单个特征或多个特征 (List[String]) 都可以
        return pd.get_dummies(df[names], prefix=prefix)
