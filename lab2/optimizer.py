import numpy as np

class AdaGrad:
    def __init__(self, alpha=3e-4, eps=1e-8):
        self.alpha = alpha
        self.eps = eps
        self.G = 0

    def optimize(self, answer, gradient):
        self.G += gradient ** 2
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
        self.v = beta * self.v + (1 - self.beta) * gradient
        self.G = self.gamma * self.G + (1 - self.gamma) * gradient ** 2
        answer -= self.alpha * gradient / np.sqrt(self.G + self.eps)
        return answer