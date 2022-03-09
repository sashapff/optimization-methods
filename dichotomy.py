class DichotomyScheduler:
    def __init__(self, f, max_lr=1, iters=10, delta=1e-2):
        self.f = f
        self.max_lr = max_lr
        self.iters = iters
        self.delta = delta
        
    def _dichotomy(self, f, a, b):
        if not 0 < 2 * self.delta < (b - a):
            raise RuntimeError("`delta` should be in [0, (a + b) / 2]")
        for iter in range(self.iters):
            x1 = (a + b) / 2 - self.delta
            x2 = (a + b) / 2 + self.delta
            if f(x1) > f(x2):
                a = x1
            else:
                b = x2
            if b - a < 2 * self.delta:
                break
        return (a + b) / 2

    def step(self, point, gradient):
        return self._dichotomy(lambda lr: self.f(point - lr * gradient), 0, self.max_lr)