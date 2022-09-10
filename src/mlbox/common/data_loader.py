"""
Feature Engineering: https://www.kaggle.com/datasets/ryanholbrook/fe-course-data
Deep Learning: https://www.kaggle.com/datasets/ryanholbrook/dl-course-data
"""
import os
import pandas as pd
from pandas import DataFrame

DIR_DATA = "/Users/lxh/workspace/kaggle-courses/data/"


class DataLoader:
    @classmethod
    def read_csv(cls, path: str, index_col: str = None, **kwargs) -> DataFrame:
        return pd.read_csv(path, index_col=index_col, **kwargs)

    @classmethod
    def load_ion(cls) -> (DataFrame, DataFrame):
        df = pd.read_csv(os.path.join(DIR_DATA, 'dl-course-data/ion.csv'), index_col=0)
        df.dropna(axis=1, inplace=True)
        X = df.drop('Class', axis=1)
        y = df['Class'].map({'good': 0, 'bad': 1})
        return X, y

    @classmethod
    def load_hotel(cls) -> (DataFrame, DataFrame):
        df = pd.read_csv(os.path.join(DIR_DATA, 'dl-course-data/hotel.csv'))
        X = df.copy()
        y = X.pop('is_canceled')
        return X, y

    @classmethod
    def load_concrete(cls) -> (DataFrame, DataFrame):
        df = pd.read_csv(os.path.join(DIR_DATA, 'dl-course-data/concrete.csv'))
        X = df.copy()
        y = X.pop("CompressiveStrength")
        return X, y

    @classmethod
    def load_spotify(cls) -> (DataFrame, DataFrame):
        df = pd.read_csv(os.path.join(DIR_DATA, 'dl-course-data/spotify.csv'))
        X = df.copy()
        y = X.pop("track_popularity")
        return X, y

    @classmethod
    def load_red_wine(cls) -> (DataFrame, DataFrame):
        df = pd.read_csv(os.path.join(DIR_DATA, 'dl-course-data/red-wine.csv'))
        X = df.copy()
        y = X.pop("quality")
        return X, y

    @classmethod
    def load_fuel(cls) -> (DataFrame, DataFrame):
        df = pd.read_csv(os.path.join(DIR_DATA, 'dl-course-data/fuel.csv'))
        X = df.copy()
        y = X.pop("FE")
        return X, y

    @classmethod
    def load_housing(cls) -> (DataFrame, DataFrame):
        df = pd.read_csv(os.path.join(DIR_DATA, 'fe-course-data/housing.csv'))
        X = df.copy()
        y = X.pop("MedHouseVal")
        return X, y

    @classmethod
    def load_ames(cls) -> (DataFrame, DataFrame):
        df = pd.read_csv(os.path.join(DIR_DATA, 'fe-course-data/ames.csv'))
        X = df.copy()
        y = X.pop("SalePrice")
        return X, y

    @classmethod
    def load_autos(cls) -> (DataFrame, DataFrame):
        df = pd.read_csv(os.path.join(DIR_DATA, 'fe-course-data/autos.csv'))
        X = df.copy()
        y = X.pop("price")
        return X, y

    @classmethod
    def load_movie_lens_1m(cls) -> (DataFrame, DataFrame):
        df = pd.read_csv(os.path.join(DIR_DATA, 'fe-course-data/movielens1m.csv'))
        X = df.copy()
        y = X.pop('Rating')
        return X, y

    @classmethod
    def load_melbourne_housing(cls) -> (DataFrame, DataFrame):
        df = pd.read_csv(os.path.join(DIR_DATA, 'competitions/melbourne-housing-snapshot/melb_data.csv'))
        X = df.copy()
        y = X.pop('Price')
        return X, y

    @classmethod
    def load_home_data(cls) -> (DataFrame, DataFrame):
        train = pd.read_csv(os.path.join(DIR_DATA, 'competitions/home-data-for-ml-course/train.csv'), index_col='Id')
        X = train.copy()
        y = X.pop('SalePrice')

        X_test = pd.read_csv(os.path.join(DIR_DATA, 'competitions/home-data-for-ml-course/test.csv'), index_col='Id')
        return X, y, X_test
