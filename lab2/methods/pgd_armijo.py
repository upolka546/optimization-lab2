"""Projected Gradient Descent with Armijo backtracking for constrained media mix."""
import numpy as np
from .base import Method, History


def project_simplex(v, budget):
    """Euclidean projection onto {x >= 0, sum(x) = budget}."""
    v = np.asarray(v, dtype=float)
    if v.ndim != 1 or v.size == 0:
        raise ValueError("v must be a non-empty 1D vector")
    if budget < 0:
        raise ValueError("budget must be non-negative")
    if budget == 0:
        return np.zeros_like(v)
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u) - budget
    idx = np.arange(1, v.size + 1)
    valid = u - cssv / idx > 0
    rho = np.flatnonzero(valid)[-1]
    theta = cssv[rho] / (rho + 1)
    return np.maximum(v - theta, 0.0)


class PGDArmijo:
    """Minimize oracle.f over the simplex using projected GD + Armijo."""

    def __init__(self, oracle, alpha0=1.0, beta=0.5, armijo_c=1e-4,
                 max_iter=10000, tol=1e-8):
        self.oracle = oracle
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        if not 0 < beta < 1 or not 0 < armijo_c < 1 or alpha0 <= 0:
            raise ValueError("Require alpha0 > 0 and beta, armijo_c in (0, 1)")
        self.alpha0, self.beta, self.armijo_c = float(alpha0), float(beta), float(armijo_c)
        self.name = "pgd_armijo"

    def run(self, x0=None):
        B = float(self.oracle.B)
        if x0 is None:
            x = np.full(self.oracle.dim, B / self.oracle.dim)
        else:
            x = project_simplex(np.asarray(x0, dtype=float), B)

        hist = History()
        f = float(self.oracle.f(x))
        g = self.oracle.grad(x)
        pg_norm = float(np.linalg.norm(x - project_simplex(x - g, B)))
        hist.append(x, f, pg_norm, 0.0)

        for _ in range(1, self.max_iter + 1):
            g = self.oracle.grad(x)
            alpha = self.alpha0
            accepted = False
            while alpha >= 1e-14:
                x_trial = project_simplex(x - alpha * g, B)
                direction = x_trial - x
                if np.linalg.norm(direction) <= self.tol:
                    accepted, f_trial = True, f
                    x_trial = x.copy()
                    break
                f_trial = float(self.oracle.f(x_trial))
                # Armijo condition for minimization along the projected direction.
                if f_trial <= f + self.armijo_c * float(g @ direction):
                    accepted = True
                    break
                alpha *= self.beta

            if not accepted:
                break
            x = x_trial
            f = float(f_trial)
            g = self.oracle.grad(x)
            # Projected-gradient mapping, not raw gradient norm, measures
            # stationarity when equality/inequality constraints are active.
            pg_norm = float(np.linalg.norm(x - project_simplex(x - g, B)))
            hist.append(x, f, pg_norm, alpha)
            if pg_norm <= self.tol:
                break
        return hist


class MediaMixConstrainedOracle:
    """Exact constrained media-mix objective using f_i(x)=a_i*x/(b_i+x).

    The optimizer minimizes -sum_i f_i(x_i); feasibility is handled by PGD,
    not by penalty terms.
    """
    def __init__(self, channels, a, b, budget=100.0):
        self.channels = list(channels)
        self.a = np.asarray(a, dtype=float)
        self.b = np.asarray(b, dtype=float)
        if self.a.ndim != 1 or self.b.shape != self.a.shape:
            raise ValueError("a and b must be equally sized 1D arrays")
        if len(self.channels) != self.a.size:
            raise ValueError("channels, a and b must have equal lengths")
        if np.any(self.a <= 0) or np.any(self.b <= 0):
            raise ValueError("All a_i and b_i must be positive")
        if budget <= 0:
            raise ValueError("budget must be positive")
        self.dim = self.a.size
        self.B = float(budget)
        self.name = f"media_mix_constrained(n={self.dim}, B={self.B:g})"

    def response(self, x):
        x = np.asarray(x, dtype=float)
        if x.shape != (self.dim,):
            raise ValueError(f"x must have shape ({self.dim},)")
        if np.any(x < -1e-10):
            raise ValueError("Response is defined for nonnegative allocations")
        x = np.maximum(x, 0.0)
        return self.a * x / (self.b + x)

    def f(self, x):
        """Minimization objective: negative total advertising response."""
        return float(-np.sum(self.response(x)))

    def grad(self, x):
        """Gradient of -sum_i a_i*x_i/(b_i+x_i)."""
        x = np.maximum(np.asarray(x, dtype=float), 0.0)
        return -self.a * self.b / np.square(self.b + x)

    def hess(self, x):
        """Diagonal Hessian; positive semidefinite on x >= 0."""
        x = np.maximum(np.asarray(x, dtype=float), 0.0)
        diagonal = 2.0 * self.a * self.b / np.power(self.b + x, 3)
        return np.diag(diagonal)
