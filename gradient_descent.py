from typing import Callable, Tuple
#%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
import random

# Function to analyze
X = np.arange(-2, 2, 0.05)
Y = np.arange(-3, 2, 0.05)
X, Y = np.meshgrid(X, Y)
Z = X**2 + Y**2

# Contour plot en 2D
plt.figure()
plt.contour(X, Y, Z, 50)


X = np.arange(-2, 2, 0.05)
Y = np.arange(-3, 2, 0.05)
X, Y = np.meshgrid(X, Y)
Z =1.5-np.exp(-X**(2)-Y**(2))-0.5*np.exp(-(X-1)**(2)-(Y+2)**(2))
# Contour plot en 2D

plt.figure()
plt.contour(X, Y, Z, 50)


class SimpleGradientDescent:
    X = np.arange(-2, 2, 0.05)
    Y = np.arange(-3, 2, 0.05)
    X, Y = np.meshgrid(X, Y)

    def __init__(self,
                 func: Callable[[float, float], float],
                 grad_func: Callable[[float, float], Tuple[float, float]],
                 alpha:float=0.1):
        self.alpha = alpha
        self.func = func
        self.grad_func = grad_func
        self.trace = None  # trace of search

    def _calc_Z_value(self):
        self.Z = self.func(self.X, self.Y)

    def plot_func(self):
        self._calc_Z_value()
        plt.figure()
        plt.contour(self.X, self.Y, self.Z, 50)
        if len(self.trace)>0:
            plt.scatter(self.trace[:,0], self.trace[:,1], s=10)

    def calculate_func_vale(self, x1:float, x2:float) -> float:
        return self.func(x1, x2)

    def calculate_func_grad(self, x1:float, x2:float) -> Tuple[float, float]:
        return self.grad_func(x1, x2)

    def gradient_descent_step(self, x1:float, x2:float) -> Tuple[float, float]:
        self.alpha = random.random()
        grad_x1, grad_x2 = self.calculate_func_grad(x1, x2)
        return (x1 - self.alpha * grad_x1, x2 - self.alpha * grad_x2)

    def minimize(self, x1_init:float, x2_init:float, steps:int, verbose:int=0, plot:bool=False)->float:
        self.trace = np.array([[x1_init, x2_init]], )
        for _ in range(steps):
            new_x1, new_x2 = self.gradient_descent_step(x1_init, x2_init)
            x1_init, x2_init = new_x1, new_x2
            self.trace = np.append(self.trace, np.array([[x1_init, x2_init]]))
        self.trace = self.trace.reshape(steps + 1, 2)
        if plot:
            self.plot_func()
        return self.func(x1_init, x2_init)


def func_f(x1: float, x2: float) -> float:
    return (x1 ** 2 + x2 ** 2)

def grad_f(x1: float, x2: float) -> Tuple[float, float]:
    return [2 * x1, 2 * x2]

x1, x2 = 1, 1   # any
sgd = SimpleGradientDescent(func_f, grad_f)
result = sgd.minimize(x1, x2, 10, verbose=0, plot=True)
print(result)
