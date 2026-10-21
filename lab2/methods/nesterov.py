# methods/nesterov.py
import numpy as np
from .base import Method


class HeavyBall(Method):
    """x_{k+1} = x_k - eta * grad f(x_k) + beta * (x_k - x_{k-1})."""

    def __init__(self, oracle, step=None, beta=0.9, max_iter=1000, tol=1e-8):
        super().__init__(oracle, max_iter=max_iter, tol=tol)
        if step is None:
            L = oracle.info.L
            step = 1.0 / L if L else 1e-3
        self.step = step
        self.beta = beta
        self.name = "heavy_ball"

    def _step(self, state):
        x = state["x"]
        x_prev = state.get("x_prev")
        if x_prev is None:               
            x_prev = x
        g = self.oracle.grad(x)
        state["x_prev"] = x
        state["x"] = x - self.step * g + self.beta * (x - x_prev)
        state["step"] = self.step
        return state


class Nesterov(Method):
    """y = x_k + beta_k (x_k - x_{k-1});  x_{k+1} = y - eta * grad f(y)."""

    def __init__(self, oracle, step=None, max_iter=1000, tol=1e-8):
        super().__init__(oracle, max_iter=max_iter, tol=tol)
        if step is None:
            L = oracle.info.L
            step = 1.0 / L if L else 1e-3
        self.step = step
        self.name = "nesterov"

    def _step(self, state):
        x = state["x"]
        x_prev = state.get("x_prev")
        if x_prev is None:
            x_prev = x
        k = state.get("nesterov_k", 0)
        beta = k / (k + 3.0)             
        y = x + beta * (x - x_prev)      
        g = self.oracle.grad(y)         
        state["x_prev"] = x
        state["x"] = y - self.step * g
        state["nesterov_k"] = k + 1
        state["step"] = self.step
        return state
