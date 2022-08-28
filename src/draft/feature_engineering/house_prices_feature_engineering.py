"""
https://www.kaggle.com/code/ryanholbrook/feature-engineering-for-house-prices

"""
import commandr
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pandas.api.types import CategoricalDtype

from category_encoders import MEstimateEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_regression
from sklearn.model_selection import KFold, cross_val_score
from xgboost import XGBRegressor

# 仅供 IPython 用
# from IPython.display import display

# Mute warnings
warnings.filterwarnings('ignore')


class DataLoader:
    # The numeric features are already encoded correctly (`float` for
    # continuous, `int` for discrete), but the categoricals we'll need to
    # do ourselves. Note in particular, that the `MSSubClass` feature is
    # read as an `int` type, but is actually a (nominative) categorical.

    # 定类 (nominal) 特征
    features_nom = [
        "MSSubClass", "MSZoning", "Street", "Alley", "LandContour", "LotConfig", "Neighborhood",
        "Condition1", "Condition2", "BldgType", "HouseStyle", "RoofStyle", "RoofMatl", "Exterior1st",
        "Exterior2nd", "MasVnrType", "Foundation", "Heating", "CentralAir", "GarageType", "MiscFeature",
        "SaleType", "SaleCondition"
    ]
    # 定序 (ordinal) 特征, Pandas 中类型为 "levels"
    five_levels = ["Po", "Fa", "TA", "Gd", "Ex"]
    ten_levels = list(range(10))

    ordered_levels = {
        "OverallQual": ten_levels,
        "OverallCond": ten_levels,
        "ExterQual": five_levels,
        "ExterCond": five_levels,
        "BsmtQual": five_levels,
        "BsmtCond": five_levels,
        "HeatingQC": five_levels,
        "KitchenQual": five_levels,
        "FireplaceQu": five_levels,
        "GarageQual": five_levels,
        "GarageCond": five_levels,
        "PoolQC": five_levels,
        "LotShape": ["Reg", "IR1", "IR2", "IR3"],
        "LandSlope": ["Sev", "Mod", "Gtl"],
        "BsmtExposure": ["No", "Mn", "Av", "Gd"],
        "BsmtFinType1": ["Unf", "LwQ", "Rec", "BLQ", "ALQ", "GLQ"],
        "BsmtFinType2": ["Unf", "LwQ", "Rec", "BLQ", "ALQ", "GLQ"],
        "Functional": ["Sal", "Sev", "Maj1", "Maj2", "Mod", "Min2", "Min1", "Typ"],
        "GarageFinish": ["Unf", "RFn", "Fin"],
        "PavedDrive": ["N", "P", "Y"],
        "Utilities": ["NoSeWa", "NoSewr", "AllPub"],
        "CentralAir": ["N", "Y"],
        "Electrical": ["Mix", "FuseP", "FuseF", "FuseA", "SBrkr"],
        "Fence": ["MnWw", "GdWo", "MnPrv", "GdPrv"],
    }

    # 对缺失值的字段补充为 None
    ordered_levels = {key: ["None"] + value for key, value in ordered_levels.items()}

    @classmethod
    def _encode(cls, df):
        # Nominal categories
        for name in cls.features_nom:
            df[name] = df[name].astype("category")
            if "None" not in df[name].cat.categories:
                df[name].cat.add_categories("None", inplace=True)
        # Ordinal categories
        for name, levels in cls.ordered_levels.items():
            df[name] = df[name].astype(CategoricalDtype(levels, ordered=True))
        return df

    @classmethod
    def _clean(cls, df):
        # 参考 ../input/house-prices-advanced-regression-techniques/data_description.txt
        # 数据中 BrkComm 命名对不上
        df["Exterior2nd"] = df["Exterior2nd"].replace({"Brk Cmn": "BrkComm"})
        # 数据裡 max(df.GarageYrBlt) == 2010, 怀疑是更新过
        # df["GarageYrBlt"] = df["GarageYrBlt"].where(df.GarageYrBlt <= 2010, df.YearBuilt)
        # 应该只是为了后面以 df.XXX 调用
        df.rename(
            columns={"1stFlrSF": "FirstFlrSF", "2ndFlrSF": "SecondFlrSF", "3SsnPorch": "ThreeSeasonPorch"},
            inplace=True)
        return df

    @classmethod
    def _impute(cls, df):
        for name in df.select_dtypes("number"):
            df[name] = df[name].fillna(0)
        for name in df.select_dtypes("category"):
            df[name] = df[name].fillna("None")
        return df

    @classmethod
    def load_data(cls):
        # 数据加載
        data_dir = Path("../input/house-prices-advanced-regression-techniques/")
        df_train = pd.read_csv(data_dir / "train.csv", index_col="Id")
        df_test = pd.read_csv(data_dir / "test.csv", index_col="Id")
        # 合并以便于预处理
        df = pd.concat([df_train, df_test])
        # 预处理
        df = cls._clean(df)
        df = cls._encode(df)
        df = cls._impute(df)
        # 按 index 重新切分 train / test
        df_train = df.loc[df_train.index, :]
        df_test = df.loc[df_test.index, :]
        return df, df_train, df_test


class MiScoresExplorer:
    def __init__(self, X, y):
        X = X.copy()
        for colname in X.select_dtypes(["object", "category"]):
            X[colname], _ = X[colname].factorize()
        # All discrete features should now have integer dtypes
        discrete_features = [pd.api.types.is_integer_dtype(t) for t in X.dtypes]
        mi_scores = mutual_info_regression(X, y, discrete_features=discrete_features, random_state=0)
        mi_scores = pd.Series(mi_scores, name="MI Scores", index=X.columns)
        mi_scores = mi_scores.sort_values(ascending=False)
        self.mi_scores = mi_scores

    def get(self):
        return self.mi_scores

    def plot(self):
        scores = self.mi_scores.sort_values(ascending=True)
        width = np.arange(len(scores))
        ticks = list(scores.index)
        plt.barh(width, scores)
        plt.yticks(width, ticks)
        plt.title("Mutual Information Scores")

    def drop_uninformative(self, df):
        return df.loc[:, self.mi_scores > 0.0]


class KMeansClustering:
    cluster_features = ["LotArea", "TotalBsmtSF", "FirstFlrSF", "SecondFlrSF", "GrLivArea"]

    def __init__(self, df, features, n_clusters=20):
        self.df = df
        self.features = features
        self.n_clusters = n_clusters

        X = self.df.copy()
        X_scaled = X.loc[:, self.features]
        self.X_scaled = (X_scaled - X_scaled.mean(axis=0)) / X_scaled.std(axis=0)
        self.kmeans = KMeans(n_clusters=self.n_clusters, n_init=50, random_state=0)

    def cluster_labels(self):
        X_new = pd.DataFrame()
        X_new["Cluster"] = self.kmeans.fit_predict(self.X_scaled)
        return X_new

    def cluster_distance(self):
        X_cd = self.kmeans.fit_transform(self.X_scaled)
        X_cd = pd.DataFrame(X_cd, columns=[f"Centroid_{i}" for i in range(X_cd.shape[1])])
        return X_cd


class PCAFeature:
    pca_features = ["GarageArea", "YearRemodAdd", "TotalBsmtSF", "GrLivArea"]

    def apply_pca(self, X, standardize=True):
        # Standardize
        if standardize:
            X = (X - X.mean(axis=0)) / X.std(axis=0)
        # Create principal components
        pca = PCA()
        X_pca = pca.fit_transform(X)
        # Convert to dataframe
        component_names = [f"PC{i + 1}" for i in range(X_pca.shape[1])]
        X_pca = pd.DataFrame(X_pca, columns=component_names)
        # Create loadings
        loadings = pd.DataFrame(
            pca.components_.T,        # transpose the matrix of loadings
            columns=component_names,  # so the columns are the principal components
            index=X.columns,          # and the rows are the original features
        )
        return pca, X_pca, loadings

    def plot_variance(self, pca, width=8, dpi=100):
        # Create figure
        fig, axs = plt.subplots(1, 2)
        n = pca.n_components_
        grid = np.arange(1, n + 1)
        # Explained variance
        evr = pca.explained_variance_ratio_
        axs[0].bar(grid, evr)
        axs[0].set(
            xlabel="Component", title="% Explained Variance", ylim=(0.0, 1.0)
        )
        # Cumulative Variance
        cv = np.cumsum(evr)
        axs[1].plot(np.r_[0, grid], np.r_[0, cv], "o-")
        axs[1].set(
            xlabel="Component", title="% Cumulative Variance", ylim=(0.0, 1.0)
        )
        # Set up figure
        fig.set(figwidth=8, dpi=100)
        return axs

    def pca_components(self, df, features):
        X = df.loc[:, features]
        _, X_pca, _ = self.apply_pca(X)
        return X_pca

    def indicate_outliers(self, df):
        # 在之前的数据分析裡發现这两个值的样本为 outliers
        # 更多 scaling 方法: https://scikit-learn.org/stable/auto_examples/preprocessing/plot_all_scaling.html
        # 更多 outlier detectors: https://scikit-learn.org/stable/modules/outlier_detection.html
        X_new = pd.DataFrame()
        X_new["Outlier"] = (df.Neighborhood == "Edwards") & (df.SaleCondition == "Partial")
        return X_new


class CrossFoldEncoder:
    """
    encoder = CrossFoldEncoder(MEstimateEncoder, m=1)
    X_encoded = encoder.fit_transform(X, y, cols=["MSSubClass"]))

    可以用其他 category_encoders 提供的 encoder
    建议了解 CatBoostEncoder: http://contrib.scikit-learn.org/category_encoders/catboost.html
    """

    def __init__(self, encoder, **kwargs):
        self.encoder_ = encoder
        self.kwargs_ = kwargs  # keyword arguments for the encoder
        self.cv_ = KFold(n_splits=5)
        self.fitted_encoders_ = None
        self.cols_ = None

    # Fit an encoder on one split and transform the feature on the
    # other. Iterating over the splits in all folds gives a complete
    # transformation. We also now have one trained encoder on each
    # fold.
    def fit_transform(self, X, y, cols):
        self.fitted_encoders_ = []
        self.cols_ = cols
        X_encoded = []
        for idx_encode, idx_train in self.cv_.split(X):
            fitted_encoder = self.encoder_(cols=cols, **self.kwargs_)
            fitted_encoder.fit(
                X.iloc[idx_encode, :], y.iloc[idx_encode],
            )
            X_encoded.append(fitted_encoder.transform(X.iloc[idx_train, :])[cols])
            self.fitted_encoders_.append(fitted_encoder)
        X_encoded = pd.concat(X_encoded)
        X_encoded.columns = [name + "_encoded" for name in X_encoded.columns]
        return X_encoded

    # To transform the test data, average the encodings learned from each fold.
    def transform(self, X):
        from functools import reduce

        X_encoded_list = []
        for fitted_encoder in self.fitted_encoders_:
            X_encoded = fitted_encoder.transform(X)
            X_encoded_list.append(X_encoded[self.cols_])
        X_encoded = reduce(
            lambda x, y: x.add(y, fill_value=0), X_encoded_list
        ) / len(X_encoded_list)
        X_encoded.columns = [name + "_encoded" for name in X_encoded.columns]
        return X_encoded


class FeatureCreator:
    """

    Interactions between the quality Qual and condition Cond features. OverallQual, for instance, was a high-scoring feature. You could try combining it with OverallCond by converting both to integer type and taking a product.
    Square roots of area features. This would convert units of square feet to just feet.
    Logarithms of numeric features. If a feature has a skewed distribution, applying a logarithm can help normalize it.
    Interactions between numeric and categorical features that describe the same thing. You could look at interactions between BsmtQual and TotalBsmtSF, for instance.
    Other group statistics in Neighboorhood. We did the median of GrLivArea. Looking at mean, std, or count could be interesting. You could also try combining the group statistics with other features. Maybe the difference of GrLivArea and the median is important?
    """
    def __init__(self, df):
        self.df = df

    @classmethod
    def label_encode(cls, df):
        X = df.copy()
        for colname in X.select_dtypes(["category"]):
            X[colname] = X[colname].cat.codes
        return X

    @classmethod
    def mathematical_transforms(cls, df):
        X = pd.DataFrame()  # dataframe to hold new features
        X["LivLotRatio"] = df.GrLivArea / df.LotArea
        X["Spaciousness"] = (df.FirstFlrSF + df.SecondFlrSF) / df.TotRmsAbvGrd
        # This feature ended up not helping performance
        # X["TotalOutsideSF"] = \
        #     df.WoodDeckSF + df.OpenPorchSF + df.EnclosedPorch + \
        #     df.ThreeSeasonPorch + df.ScreenPorch
        return X

    @classmethod
    def interactions(cls, df):
        X = pd.get_dummies(df.BldgType, prefix="Bldg")
        X = X.mul(df.GrLivArea, axis=0)
        return X

    @classmethod
    def counts(cls, df):
        X = pd.DataFrame()
        X["PorchTypes"] = df[
            ["WoodDeckSF", "OpenPorchSF", "EnclosedPorch", "ThreeSeasonPorch", "ScreenPorch"]
        ].gt(0.0).sum(axis=1)
        return X

    @classmethod
    def break_down(cls, df):
        X = pd.DataFrame()
        X["MSClass"] = df.MSSubClass.str.split("_", n=1, expand=True)[0]
        return X

    @classmethod
    def group_transforms(cls, df):
        X = pd.DataFrame()
        X["MedNhbdArea"] = df.groupby("Neighborhood")["GrLivArea"].transform("median")
        return X


class Course:
    @staticmethod
    def init_plt():
        plt.style.use("seaborn-whitegrid")
        plt.rc("figure", autolayout=True)
        plt.rc("axes", labelweight="bold",  labelsize="large", titleweight="bold", titlesize=14, titlepad=10)

    @staticmethod
    def score_dataset(X, y, model=XGBRegressor()):
        for colname in X.select_dtypes(["category"]):
            X[colname] = X[colname].cat.codes
        log_y = np.log(y)
        score = cross_val_score(model, X, log_y, cv=5, scoring="neg_mean_squared_error")
        return - np.sqrt(score.mean())

    @staticmethod
    def corrplot(df, method="pearson", annot=True, **kwargs):
        sns.clustermap(
            df.corr(method),
            vmin=-1.0,
            vmax=1.0,
            cmap="icefire",
            method="complete",
            annot=annot,
            **kwargs,
        )


def find_best_params_for_xbg(X_train, y_train):
    """
    此样例用于为 XGB Regressor 自动发现最佳超参数
    Optuna's visualizations
    https://optuna.readthedocs.io/en/stable/tutorial/10_key_features/005_visualization.html
    """
    import optuna

    def score_dataset(X, y, model=XGBRegressor()):
        for colname in X.select_dtypes(["category"]):
            X[colname] = X[colname].cat.codes
        log_y = np.log(y)
        score = cross_val_score(model, X, log_y, cv=5, scoring="neg_mean_squared_error")
        return - np.sqrt(score.mean())

    def objective(trial):
        xgb_params = dict(
            max_depth=trial.suggest_int("max_depth", 2, 10),
            learning_rate=trial.suggest_float("learning_rate", 1e-4, 1e-1, log=True),
            n_estimators=trial.suggest_int("n_estimators", 1000, 8000),
            min_child_weight=trial.suggest_int("min_child_weight", 1, 10),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.2, 1.0),
            subsample=trial.suggest_float("subsample", 0.2, 1.0),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-4, 1e2, log=True),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-4, 1e2, log=True),
        )
        xgb = XGBRegressor(**xgb_params)
        return score_dataset(X_train, y_train, xgb)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=20)
    xgb_params = study.best_params
    return xgb_params


def create_features(df, df_test=None):
    X = df.copy()
    y = X.pop("SalePrice")

    mi_scores_explorer = MiScoresExplorer(X, y)
    mi_scores = mi_scores_explorer.get()

    # Combine splits if test data is given
    #
    # If we're creating features for test set predictions, we should
    # use all the data we have available. After creating our features,
    # we'll recreate the splits.
    if df_test is not None:
        X_test = df_test.copy()
        X_test.pop("SalePrice")
        X = pd.concat([X, X_test])

    # Lesson 2 - Mutual Information
    mi_scores_explorer = MiScoresExplorer(X, y)
    X = mi_scores_explorer.drop_uninformative(X)

    # Lesson 3 - Transformations
    X = X.join(FeatureCreator.mathematical_transforms(X))
    X = X.join(FeatureCreator.interactions(X))
    X = X.join(FeatureCreator.counts(X))
    # X = X.join(break_down(X))
    X = X.join(FeatureCreator.group_transforms(X))

    # Lesson 4 - Clustering
    # X = X.join(cluster_labels(X, cluster_features, n_clusters=20))
    # X = X.join(cluster_distance(X, cluster_features, n_clusters=20))

    # Lesson 5 - PCA
    pca_inspired = pd.DataFrame()
    pca_inspired["Feature1"] = X.GrLivArea + X.TotalBsmtSF
    pca_inspired["Feature2"] = X.YearRemodAdd * X.TotalBsmtSF
    X = X.join(pca_inspired)
    # X = X.join(pca_components(X, pca_features))
    # X = X.join(indicate_outliers(X))

    X = FeatureCreator.label_encode(X)

    # Reform splits
    if df_test is not None:
        X_test = X.loc[df_test.index, :]
        X.drop(df_test.index, inplace=True)

    # Lesson 6 - Target Encoder
    encoder = CrossFoldEncoder(MEstimateEncoder, m=1)
    X = X.join(encoder.fit_transform(X, y, cols=["MSSubClass"]))
    if df_test is not None:
        X_test = X_test.join(encoder.transform(X_test))

    if df_test is not None:
        return X, X_test
    else:
        return X


@commandr.command
def main():
    Course.init_plt()
    df, df_train, df_test = DataLoader.load_data()

    X = df_train.copy()
    y = X.pop("SalePrice")
    baseline_score = Course.score_dataset(X, y)
    print(baseline_score)

    mi_scores_explorer = MiScoresExplorer(X, y)
    mi_scores = mi_scores_explorer.get()
    X = mi_scores_explorer.drop_uninformative(X)
    score = Course.score_dataset(X, y)
    print(score)

    X_train = create_features(df_train)
    y_train = df_train.loc[:, "SalePrice"]
    Course.score_dataset(X_train, y_train)

    X_train = create_features(df_train)
    y_train = df_train.loc[:, "SalePrice"]

    xgb_params = dict(
        max_depth=6,           # maximum depth of each tree - try 2 to 10
        learning_rate=0.01,    # effect of each tree - try 0.0001 to 0.1
        n_estimators=1000,     # number of trees (that is, boosting rounds) - try 1000 to 8000
        min_child_weight=1,    # minimum number of houses in a leaf - try 1 to 10
        colsample_bytree=0.7,  # fraction of features (columns) per tree - try 0.2 to 1.0
        subsample=0.7,         # fraction of instances (rows) per tree - try 0.2 to 1.0
        reg_alpha=0.5,         # L1 regularization (like LASSO) - try 0.0 to 10.0
        reg_lambda=1.0,        # L2 regularization (like Ridge) - try 0.0 to 10.0
        num_parallel_tree=1,   # set > 1 for boosted random forests
    )

    xgb = XGBRegressor(**xgb_params)
    Course.score_dataset(X_train, y_train, xgb)


if __name__ == '__main__':
    commandr.Run()
