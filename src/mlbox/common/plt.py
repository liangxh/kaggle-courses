import matplotlib.pyplot as plt


class Plot:
    @classmethod
    def init(cls):
        plt.style.use("seaborn-whitegrid")
        plt.rc("figure", autolayout=True)
        plt.rc("axes", labelweight="bold",  labelsize="large", titleweight="bold", titlesize=14, titlepad=10)
