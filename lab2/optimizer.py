import numpy as np


class BaseOptimizer:
    def __init__(self, lr=1e-3, reg_coef=1e-1):
        self.lr = lr
        self.reg_coef = reg_coef

    def optimize(self, answer, gradient):
        answer -= self.lr * gradient + self.reg_coef * answer
        return answer


class Momentum:
    def __init__(self, alpha=3e-4, beta=0.9):
        self.alpha = alpha
        self.beta = beta
        self.diff = 0

    def optimize(self, answer, gradient):
        self.diff = self.alpha * self.diff - self.beta * gradient
        answer += self.diff
        return answer


class Nesterov:
    def __init__(self, alpha=3e-4, beta=0.9):
        self.alpha = alpha
        self.beta = beta
        self.diff = 0

    def optimize(self, answer, gradient):
        diff_prev = self.diff
        self.diff = answer - self.beta * gradient
        answer = self.diff + self.alpha * (self.diff - diff_prev)
        return answer


class AdaGrad:
    def __init__(self, alpha=3e-4, eps=1e-8):
        self.alpha = alpha
        self.eps = eps
        self.G = 0

    def optimize(self, answer, gradient):
        self.G += np.multiply(gradient, gradient)
        answer -= self.alpha * gradient / np.sqrt(self.G + self.eps)
        return answer


class RMSProp:
    def __init__(self, alpha=3e-4, gamma=0.99, eps=1e-8):
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps
        self.G = 0

    def optimize(self, answer, gradient):
        self.G = self.G * self.gamma + (1 - self.gamma) * gradient ** 2
        answer -= self.alpha * gradient / np.sqrt(self.G + self.eps)
        return answer


class Adam:
    def __init__(self, alpha=3e-4, beta=0.9, gamma=0.99, eps=1e-8):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.eps = eps
        self.G = 0
        self.v = 0

    def optimize(self, answer, gradient):
        self.v = self.beta * self.v + (1 - self.beta) * gradient
        self.G = self.gamma * self.G + (1 - self.gamma) * gradient ** 2
        answer -= self.alpha * gradient / np.sqrt(self.G + self.eps)
        return answer
