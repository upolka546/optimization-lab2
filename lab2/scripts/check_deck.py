# scripts/check_deck.py  -- проверка чисел со слайдов 9-14 на ваших методах
import numpy as np
from oracles.base import Oracle, OracleInfo
from methods import GD, Nesterov, HeavyBall


class Deck(Oracle):
    """Задача со слайдов: a, b со слайда 4; B=100, lam=0.3, mu=1, rho=10, alpha=0.005."""

    def __init__(self, mu=1.0, alpha=0.005, lam=0.3, rho=10.0, B=100.0):
        self.a = np.array([180, 150, 90, 120, 100, 60.0])
        self.b = np.array([40, 25, 30, 10, 15, 8.0])
        self.mu, self.alpha, self.lam, self.rho, self.B = mu, alpha, lam, rho, B
        self.dim = 6
        self.name = "deck"

    def _r(self, x):
        xp = np.maximum(x, 0)
        return np.where(x >= 0, self.a * xp / (self.b + xp), self.a / self.b * x)

    def _dr(self, x):
        xp = np.maximum(x, 0)
        return np.where(x >= 0, self.a * self.b / (self.b + xp) ** 2, self.a / self.b)

    def _ddr(self, x):
        xp = np.maximum(x, 0)
        return np.where(x >= 0, -2 * self.a * self.b / (self.b + xp) ** 3, 0.0)

    def f(self, x):
        return float(-self._r(x).sum() + self.lam * x.sum()
                     + 0.5 * self.mu * (x.sum() - self.B) ** 2
                     + 0.5 * self.rho * np.sum(np.minimum(x, 0) ** 2)
                     + 0.5 * self.alpha * x @ x)

    def grad(self, x):
        return (-self._dr(x) + self.lam + self.mu * (x.sum() - self.B)
                + self.rho * np.minimum(x, 0) + self.alpha * x)

    def hess(self, x):
        return (np.diag(-self._ddr(x) + self.rho * (x < 0) + self.alpha)
                + self.mu * np.ones((6, 6)))

    @property
    def info(self):
        L = float(np.max(2 * self.a / self.b ** 2) + 6 * self.mu + self.alpha)
        return OracleInfo(L=L, mu=self.alpha)


o = Deck()
x0 = np.zeros(6)
print("L =", round(o.info.L, 3))
for name, m in [("GD, шаг 1/L", GD(o, max_iter=100000, tol=1e-6)),
                ("Nesterov, шаг 1/L", Nesterov(o, max_iter=100000, tol=1e-6)),
                ("Heavy ball b=0.9, шаг 1/L", HeavyBall(o, beta=0.9, max_iter=100000, tol=1e-6))]:
    h = m.run(x0)
    print(f"{name:28s} итераций: {len(h.f) - 1}")
