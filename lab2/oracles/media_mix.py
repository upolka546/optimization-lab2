import numpy as np
from .base import Oracle, OracleInfo

# Синтетические каналы: (название, a, b); a — максимальный отклик, b — полунасыщение
_CHANNELS = {
    "TV": (60.0, 40.0),
    "Digital": (35.0, 8.0),
    "OOH": (15.0, 10.0),
    "Search": (25.0, 3.0),
    "Social": (30.0, 6.0),
    "Radio": (6.0, 0.5),
}
_ORDER_6 = ["TV", "Digital", "OOH", "Search", "Social", "Radio"]
_ORDER_2 = ["Search", "TV"]


class MediaMixOracle(Oracle):
    """Оптимизация медиамикса (бюджет заменён штрафом).

    F(x) = - sum r_i(x_i) + rho/2 (sum x - B)^2
           + gamma/2 sum max(0, -x_i)^2 + lam/2 sum (x_i - xbar_i)^2

    r_i(x) = a_i x / (b_i + x) (функция Хилла) при x >= 0. У Хилла полюс в
    x = -b, поэтому при x < 0 берём C1-продолжение r_i(x) = (a_i/b_i) x
    (то же значение и наклон в нуле). Тогда F выпукла и конечна всюду.
    xbar — прошлогодний план (равное деление бюджета).
    """

    def __init__(self, n_channels: int = 6, budget: float = 100.0,
                 rho: float = 1.0, gamma: float = 10.0, lam: float = 0.01,
                 seed: int = 0, noise: float = 0.0):
        n = n_channels
        rng = np.random.default_rng(seed)
        if n == 6:
            self.channels = list(_ORDER_6)
        elif n == 2:
            self.channels = list(_ORDER_2)
        else:
            self.channels = None
        if self.channels is not None:
            self.a = np.array([_CHANNELS[c][0] for c in self.channels])
            self.b = np.array([_CHANNELS[c][1] for c in self.channels])
        else:
            self.channels = [f"ch{i}" for i in range(n)]
            self.b = np.exp(rng.uniform(np.log(0.5), np.log(50.0), n))
            self.a = rng.uniform(5.0, 60.0, n)
        if noise > 0:
            self.a = self.a * np.maximum(0.1, 1.0 + noise * rng.standard_normal(n))

        self.dim = n
        self.B = float(budget)
        self.rho, self.gamma, self.lam = rho, gamma, lam
        self.xbar = np.full(n, self.B / n)
        self.name = f"media_mix(n={n}, B={budget:g}, rho={rho:g}, lam={lam:g})"
        self._x_star, self._f_star = self._solve()

    # ---------- отклик ----------
    def response(self, x):
        """Отклик r_i(x_i) по каналам."""
        return self.response_a(x, self.a)

    def _dr(self, x):
        return self.dr_a(x, self.a)

    def _ddr(self, x):
        return self.ddr_a(x, self.a)

    # варианты с произвольным a (вектор n или матрица S x n) — для робастной версии
    def response_a(self, x, a):
        x = np.asarray(x, dtype=float)
        xp = np.maximum(x, 0.0)
        return np.where(x >= 0, a * xp / (self.b + xp), a / self.b * x)

    def dr_a(self, x, a):
        xp = np.maximum(x, 0.0)
        return np.where(x >= 0, a * self.b / (self.b + xp) ** 2, a / self.b)

    def ddr_a(self, x, a):
        xp = np.maximum(x, 0.0)
        return np.where(x >= 0, -2 * a * self.b / (self.b + xp) ** 3, 0.0)

    # ---------- оракул ----------
    def f(self, x: np.ndarray) -> float:
        s = np.sum(x) - self.B
        return float(-np.sum(self.response(x)) + 0.5 * self.rho * s * s
                     + 0.5 * self.gamma * np.sum(np.maximum(0.0, -x) ** 2)
                     + 0.5 * self.lam * np.sum((x - self.xbar) ** 2))

    def grad(self, x: np.ndarray) -> np.ndarray:
        s = np.sum(x) - self.B
        return (-self._dr(x) + self.rho * s + self.gamma * np.minimum(x, 0.0)
                + self.lam * (x - self.xbar))

    def _diag(self, x):
        return -self._ddr(x) + self.gamma * (x < 0) + self.lam

    def hess(self, x: np.ndarray) -> np.ndarray:
        return np.diag(self._diag(x)) + self.rho * np.ones((self.dim, self.dim))

    def hess_diag(self, x: np.ndarray) -> np.ndarray:
        return self._diag(x) + self.rho

    def hess_vec(self, x: np.ndarray, v: np.ndarray) -> np.ndarray:
        # диагональ + ранг один: O(n)
        return self._diag(x) * v + self.rho * np.sum(v)

    def local_condition(self, x: np.ndarray):
        """(lam_min, lam_max, kappa) гессиана в точке x."""
        e = np.linalg.eigvalsh(self.hess(x))
        return float(e[0]), float(e[-1]), float(e[-1] / e[0])

    # ---------- решение ----------
    def _solve(self):
        """Метод Ньютона с backtracking; проверка через scipy."""
        x = self.xbar.copy()
        for _ in range(100):
            g = self.grad(x)
            if np.linalg.norm(g) < 1e-12:
                break
            p = -np.linalg.solve(self.hess(x), g)
            t, f0 = 1.0, self.f(x)
            while self.f(x + t * p) > f0 + 1e-4 * t * (g @ p) and t > 1e-12:
                t *= 0.5
            x = x + t * p
        f_star = self.f(x)
        from scipy.optimize import minimize
        res = minimize(self.f, self.xbar, jac=self.grad, method="trust-exact",
                       hess=self.hess, options={"gtol": 1e-12})
        assert np.allclose(res.x, x, rtol=1e-6, atol=1e-6) and abs(res.fun - f_star) < 1e-9, \
            f"Newton и scipy расходятся: {np.max(np.abs(res.x - x)):.2e}"
        return x, f_star

    @property
    def info(self) -> OracleInfo:
        # Кривизна Хилла максимальна в x=0: -r'' = 2a/b^2; при x<0 даёт gamma.
        # Штраф бюджета — rank-one rho*11^T с собственным значением n*rho.
        L = float(np.max(np.maximum(2 * self.a / self.b ** 2, self.gamma))
                  + self.dim * self.rho + self.lam)
        return OracleInfo(L=L, mu=self.lam, x_star=self._x_star.copy(),
                          f_star=self._f_star)
