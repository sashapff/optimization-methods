import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib


class GradientDescent:
    def __init__(self,
                 function=lambda x: x[0] ** 2,
                 derivative=lambda x: 2 * x[0],
                 iterations=100,
                 initial_point=np.ndarray([1]),
                 epsilon=1e-3,
                 scheduler=None,
                 scheduler_params=None):
        self.derivative = derivative
        self.function = function
        self.max_iterations_count = iterations
        self.initial_point = initial_point
        self.epsilon = epsilon
        self.trace_points = []
        self.trace_function_results = []
        self.processed_iterations_count = 0
        self.scheduler = scheduler
        self.scheduler_params = scheduler_params

    def plot_trace_3d(self, title):
        raise NotImplementedError()

    def plot_trace_2d(self, title):
        sns.scatterplot(x=[e[0] for e in self.trace_points], y=[e[0] for e in self.trace_function_results],
                        color='red').set_title(title)
        plt.show()

    def plot_function_2d(self, from_x=-1, to_x=1):
        xs = []
        ys = []
        for x in np.arange(from_x, to_x, 0.01):
            xs.append(x)
            ys.append(self.function(x))
        sns.lineplot(x=xs, y=ys)

    def animation_function(i):
        pass

    def plot_trace(self, title='', animate=False, with_function=False, from_x=-1, to_x=1):
        if self.initial_point.shape[0] == 1:
            if with_function:
                self.plot_function_2d(from_x, to_x)
            if animate:
                pass
                # ani = matplotlib.animation.FuncAnimation(fig, animate,
                #                                          frames=len(self.trace_points),
                #                                          repeat=True)
            else:
                self.plot_trace_2d(title)
        elif self.initial_point.shape[0] == 2:
            self.plot_trace_3d(title)
        else:
            raise NotImplementedError("Not supported function")

    def optimize(self):
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

            next_point = current_point - lr * cur_gradient_result
            similar = True
            for old_p, new_p in zip(next_point, current_point):
                if abs(new_p - old_p) > self.epsilon:
                    similar = False
            if similar:
                break
            current_point = next_point
        self.optimized_minimum = current_point
        self.optimized_iterations = iteration + 1
        return current_point, iteration + 1
