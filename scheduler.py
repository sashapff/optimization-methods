class ConstLRScheduler:
    def __init__(self, lr):
        self.lr = lr

    def step(self):
        return self.lr


class ExpLRScheduler:
    def __init__(self, lr=0.1, period=10, gamma=0.1):
        self.lr = lr
        self.period = period
        self.gamma = gamma
        self.step_count = 0

    def step(self):
        self.step_count += 1
        if self.step_count % self.period == 0:
            self.step_count = 0
            self.lr *= self.gamma
        return self.lr
