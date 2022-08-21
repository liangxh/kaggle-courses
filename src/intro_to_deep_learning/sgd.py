"""
https://www.kaggle.com/code/ryanholbrook/stochastic-gradient-descent/tutorialfuel.csv
"""
import commandr
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer, make_column_selector
from sklearn.model_selection import train_test_split

# Setup plotting
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
# Set Matplotlib defaults
plt.rc('figure', autolayout=True)
plt.rc('axes', labelweight='bold', labelsize='large',
       titleweight='bold', titlesize=18, titlepad=10)
plt.rc('animation', html='html5')


@commandr.command
def fuel_economy_demo():
    # 读取数据
    fuel = pd.read_csv('../../input/dl-course-data/fuel.csv')

    # 生成训练样本
    X = fuel.copy()
    y = X.pop('FE')

    # 预处理器
    preprocessor = make_column_transformer(
        (StandardScaler(), make_column_selector(dtype_include=np.number)),
        (OneHotEncoder(sparse=False), make_column_selector(dtype_include=object)),
    )

    # 预处理
    X = preprocessor.fit_transform(X)
    y = np.log(y)  # log transform target instead of standardizing

    input_shape = [X.shape[1]]

    # 模型搭建
    model = keras.Sequential([
        layers.Dense(128, activation='relu', input_shape=input_shape),
        layers.Dense(128, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(1),
    ])

    model.compile(optimizer="adam", loss="mae")

    history = model.fit(
        X, y,
        validation_data=(X, y),
        batch_size=256,
        epochs=10,
    )

    # convert the training history to a dataframe
    history_df = pd.DataFrame(history.history)
    # use Pandas native plot method
    history_df.loc[5:, ['loss']].plot()


if __name__ == '__main__':
    # commandr.Run()
    fuel_economy_demo()
