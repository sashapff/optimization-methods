import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from IPython.display import HTML
from celluloid import Camera


class GradientDescent:
    def __init__(self,
                 function=lambda x: x ** 2,
                 derivative=lambda x: 2 * x,
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

    def plot_trace_3d(self, title):
        raise NotImplementedError()

    def plot_trace_2d(self, title):
        sns.scatterplot(x=[e[0] for e in self.trace_points], y=[e[0] for e in self.trace_function_results],
                        color='maroon').set_title(title)
        plt.show()

    def plot_function_2d(self, from_x=-1, to_x=1):
        xs = []
        ys = []
        for x in np.arange(from_x, to_x, 0.01):
            xs.append(x)
            ys.append(self.function(x))
        return sns.lineplot(x=xs, y=ys, color='skyblue', lw=0.75)

    def animation_function(self, i, plot_one_point=False):
        if plot_one_point:
            sns.scatterplot(x=[e[0] for e in [self.trace_points[i]]],
                        y=[e[0] for e in [self.trace_function_results[i]]],
                        color='red')
        else:
            sns.scatterplot(x=[e[0] for e in self.trace_points[:i + 1]],
                        y=[e[0] for e in self.trace_function_results[:i + 1]],
                        color='red')

    def plot_trace(self, title='', animate=False, with_function=False, from_x=-1, to_x=1, filename=None, plot_one_point=False):
        if self.initial_point.shape[0] == 1:
            fig, ax = plt.subplots()
            if animate:
                camera = Camera(fig)
                fig.suptitle(title)
                for i in range(len(self.trace_points)):
                    ax.text(0.2, 0.95, f'iteration: {i}', transform=ax.transAxes)
                    self.plot_function_2d(from_x, to_x)
                    self.animation_function(i, plot_one_point)
                    camera.snap()
                animation = camera.animate(interval=2000//len(self.trace_points))
                if filename is None:
                    filename = 'ani.gif'
                animation.save(filename, fps=int(2000/len(self.trace_points)))
                plt.close(fig)
                return HTML(f'<img src="{filename}">')
            else:
                if with_function:
                    self.plot_function_2d(from_x, to_x)
                self.plot_trace_2d(title)
                plt.close(fig)
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
                lr = self.scheduler.step(*(current_point, cur_gradient_result))
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
