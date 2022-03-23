import random
import time
from copy import copy
import seaborn as sns
import numpy as np


class BaseOptimizer:
    def __init__(self, lr, reg_coef):
        self.lr = lr
        self.reg_coef = reg_coef

    def optimize(self, answer, gradient):
        answer -= self.lr * gradient + self.reg_coef * answer
        return answer


class SGDRegressor:
    def __init__(self, epochs_count=100000, lr=1e-3, reg_coef=1, batch_size=1, optimizer=BaseOptimizer(1e-3, 1)):
        self.epochs_count = epochs_count
        self.lr = lr
        self.reg_coef = reg_coef
        self.batch_size = batch_size
        self.best_error = 1e25
        self.error_log = []
        self.best_params = []
        self.optimizer = optimizer

    def fit(self, X_train, y_train, process_steps=True, plot_errors=True):
        self.X_train = np.append(X_train, np.ones([X_train.shape[0], 1]), axis=1)
        self.params_count = self.X_train.shape[0]

        self.y_train = y_train
        self.batch_size = min(self.batch_size, y_train.shape[0])

        self.answer = np.sqrt(self.params_count) * np.random.randn(1, self.params_count)
        if process_steps:
            for i in range(self.epochs_count):
                self.process_step()
            if plot_errors:
                self.plot_errors_log()
            return self.answer

    def plot_errors_log(self):
        sns.scatterplot(x=[i for i in range(len(self.error_log))], y=self.error_log)

    def process_step(self):
        stochastic_index = np.random.randint(self.X_train.shape[0], size=self.batch_size).astype(int)

        stochastic_elements = self.X_train[stochastic_index, :]

        y_predicted = self.predict(stochastic_elements)
        error = y_predicted - self.y_train[stochastic_index]
        mean_error = np.mean(np.abs(error))
        self.error_log.append(mean_error)

        if abs(mean_error) < self.best_error:
            self.best_error = abs(mean_error)
            self.best_params = self.answer
        gradient = 2 * stochastic_elements.T @ error
        self.answer = self.optimizer.optimize(self.answer, gradient)

    def predict(self, X_test):
        return np.squeeze(X_test @ self.answer.T, axis=1)
        # return [sum(w * x_st for w, x_st in zip(self.answer, element)) for element in X_test]


if __name__ == "__main__":
    n, m = map(int, input().split())
    X_train, y_train = [], []
    for i in range(n):
        obj = list(map(int, input().split()))
        X_train.append(obj[:-1])
        y_train.append(obj[-1])
    sgd = SGDRegressor(40000, 0.05, 0, 2)
    for i in sgd.fit(np.array(X_train), np.array(y_train)):
        print(i)
