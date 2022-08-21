import commandr
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

plt.style.use('seaborn-whitegrid')
# Set Matplotlib defaults
plt.rc('figure', autolayout=True)
plt.rc('axes', labelweight='bold', labelsize='large',
       titleweight='bold', titlesize=18, titlepad=10)


class SequentialDemo:
    @staticmethod
    def build_model():
        model = keras.Sequential([
            # the hidden ReLU layers
            layers.Dense(units=4, activation='relu', input_shape=[2]),
            layers.Dense(units=3, activation='relu'),
            # the linear output layer
            layers.Dense(units=1),
        ])

    @staticmethod
    def test_activation(x):
        activation_layer = layers.Activation('relu')
        y = activation_layer(x)
        return y


@commandr.command('act')
def test_activation(x):
    x = float(x)
    print(SequentialDemo.test_activation(x))


@commandr.command('act_demo')
def test_activation_graph():
    # 绘制 relu 曲线
    activation_layer = layers.Activation('relu')

    x = tf.linspace(-3.0, 3.0, 100)
    y = activation_layer(x)

    plt.figure(dpi=100)
    plt.plot(x, y)
    plt.xlim(-3, 3)
    plt.xlabel("Input")
    plt.ylabel("Output")
    plt.show()


if __name__ == '__main__':
    commandr.Run()
