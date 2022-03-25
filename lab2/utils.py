import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D


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


def plot_trace_3d(title, weights_log, ax3d):
    step = max(1, len(weights_log) // 150)
    X = [weights_log[w][0][0] for w in range(0, len(weights_log), step)]
    Y = [weights_log[w][0][1] for w in range(0, len(weights_log), step)]
    Z = [weights_log[w][0][2] for w in range(0, len(weights_log), step)]

    ax3d.plot(X, Y, Z, '-o', label=title)
