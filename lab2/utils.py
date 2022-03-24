import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def read_data():
    X = pd.read_csv("X.csv").to_numpy()[:, 1:]
    y = pd.read_csv("y.csv").to_numpy()[:, 1]
    return X, y


def plot_log(data, title):
    sns.lineplot(x=[i for i in range(len(data))], y=data)
    plt.title(title)
    plt.show()


def min_max_scaling(X):
    return (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
    

def standard_scaling(X):
    return (X - X.mean(axis=0)) / X.std(axis=0, ddof=1)