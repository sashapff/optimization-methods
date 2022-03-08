import numpy as np


class GradientDescent:
    def __init__(self,
                 function=lambda x: x[0] ** 2,
                 derivative=lambda x: 2 * x[0],
                 iterations=100,
                 initial_point=np.ndarray([1]),
                 epsilon=1e-3,
                 scheduler=None):
        self.derivative = derivative
        self.function = function
        self.max_iterations_count = iterations
        self.initial_point = initial_point
        self.epsilon = epsilon
        self.trace_points = []
        self.trace_function_results = []
        self.processed_iterations_count = 0
        self.scheduler = scheduler

    def optimize(self):
        previous_point = self.initial_point
        current_point = self.initial_point
        iteration = 0
        for iteration in range(self.max_iterations_count):
            cur_function_result = self.function(current_point)
            cur_gradient_result = self.derivative(current_point)

            self.trace_points.append(current_point)
            self.trace_function_results.append(cur_function_result)

            if self.scheduler is not None:
                lr = self.scheduler.step()
            else:
                lr = 0.001

            previous_point = current_point
            current_point = current_point - lr * cur_gradient_result

            similar = True
            for old_p, new_p in zip(previous_point, current_point):
                if abs(new_p - old_p) > self.epsilon:
                    similar = False
            if similar:
                break

        return current_point, iteration + 1
