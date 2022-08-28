from pandas import DataFrame
from pandas.api.types import CategoricalDtype
from typing import List, Dict


class DataFramePreprocessor:
    @classmethod
    def replace_value(cls, df: DataFrame,
                      name: str, value_mapping: Dict[str, str]):
        df[name] = df[name].replace(value_mapping)

    @classmethod
    def rename(cls, df: DataFrame,
               name: str, new_name: str, inplace: bool = False):
        df.rename(columns={name: new_name}, inplace=inplace)

    @classmethod
    def encode_categorical_features(cls, df: DataFrame,
                                    names: List[str], value_none: any = "None"):
        for name in names:
            df[name] = df[name].astype("category")
            if value_none not in df[name].cat.categories:
                df[name].cat.add_categories("None", inplace=True)

    @classmethod
    def encode_ordinal_features(cls, df: DataFrame,
                                name: str, levels: List[any], value_none: any = "None"):
        if value_none not in levels:
            levels = [value_none] + levels
        df[name] = df[name].astype(CategoricalDtype(levels, ordered=True))

    @classmethod
    def fillna(cls, df: DataFrame,
               num_default: int = 0, category_default: str = "None"):
        for name in df.select_dtypes("number"):
            df[name] = df[name].fillna(num_default)

        for name in df.select_dtypes("category"):
            df[name] = df[name].fillna(category_default)

