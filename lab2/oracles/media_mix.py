import numpy as np
from .base import Oracle, OracleInfo

# Синтетические каналы: (название, a, b); a — максимальный отклик, b — полунасыщение
_CHANNELS = {
    "TV": (180.0, 40.0),
    "Digital": (150.0, 25.0),
    "OOH": (90.0, 30.0),
    "Search": (120.0, 10.0),
    "Social": (100.0, 15.0),
    "Radio": (60.0, 8.0),
}
_ORDER_6 = ["TV", "Digital", "OOH", "Search", "Social", "Radio"]
_ORDER_2 = ["Search", "TV"]


class MediaMixOracle(Oracle):
    """Оптимизация медиамикса (бюджет заменён штрафом).

    F(x) = - sum_i f_i(x_i) + mu/2 (sum x_i - B)^2
           + rho/2 sum min(x_i, 0)^2 + alpha/2 ||x||^2

    f_i(x) = a_i x / (b_i + x) (функция Хилла, h=1) при x >= 0. У Хилла полюс
    в x = -b, поэтому при x < 0 берём C1-продолжение f_i(x) = (a_i/b_i) x
    (то же значение и наклон в нуле). Тогда F выпукла и конечна всюду.
    mu — штраф за отклонение суммы от бюджета, rho — штраф за отрицательные
    x_i, alpha — L2-регуляризация к нулю (сильная выпуклость).
    """

    def __init__(self, n_channels: int = 6, budget: float = 100.0,
                 mu: float = 1.0, rho: float = 10.0, alpha: float = 0.005,
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
        self.mu, self.rho, self.alpha = mu, rho, alpha
        # стартовая точка Ньютона (равное деление бюджета); в F не входит
        self.x_start = np.full(n, self.B / n)
        self.name = f"media_mix(n={n}, B={budget:g}, mu={mu:g}, rho={rho:g}, alpha={alpha:g})"
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
        return float(-np.sum(self.response(x)) + 0.5 * self.mu * s * s
                     + 0.5 * self.rho * np.sum(np.minimum(x, 0.0) ** 2)
                     + 0.5 * self.alpha * np.sum(x * x))

    def grad(self, x: np.ndarray) -> np.ndarray:
        s = np.sum(x) - self.B
        return (-self._dr(x) + self.mu * s + self.rho * np.minimum(x, 0.0)
                + self.alpha * x)

    def _diag(self, x):
        return -self._ddr(x) + self.rho * (x < 0) + self.alpha

    def hess(self, x: np.ndarray) -> np.ndarray:
        return np.diag(self._diag(x)) + self.mu * np.ones((self.dim, self.dim))

    def hess_diag(self, x: np.ndarray) -> np.ndarray:
        return self._diag(x) + self.mu

    def hess_vec(self, x: np.ndarray, v: np.ndarray) -> np.ndarray:
        # диагональ + ранг один: O(n)
        return self._diag(x) * v + self.mu * np.sum(v)

    def local_condition(self, x: np.ndarray):
        """(lam_min, lam_max, kappa) гессиана в точке x."""
        e = np.linalg.eigvalsh(self.hess(x))
        return float(e[0]), float(e[-1]), float(e[-1] / e[0])

    # ---------- решение ----------
    def _solve(self):
        """Метод Ньютона с backtracking; проверка через scipy."""
        x = self.x_start.copy()
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
        res = minimize(self.f, self.x_start, jac=self.grad, method="trust-exact",
                       hess=self.hess, options={"gtol": 1e-12})
        assert np.allclose(res.x, x, rtol=1e-6, atol=1e-6) and abs(res.fun - f_star) < 1e-9, \
            f"Newton и scipy расходятся: {np.max(np.abs(res.x - x)):.2e}"
        return x, f_star

    @property
    def info(self) -> OracleInfo:
        # Кривизна Хилла максимальна в x=0: -f'' = 2a/b^2; при x<0 даёт rho.
        # Штраф бюджета — rank-one mu*11^T с собственным значением n*mu.
        # ВНИМАНИЕ: в OracleInfo поле `mu` = константа сильной выпуклости
        # (это alpha), а НЕ штраф бюджета self.mu.
        L = float(np.max(np.maximum(2 * self.a / self.b ** 2, self.rho))
                  + self.dim * self.mu + self.alpha)
        return OracleInfo(L=L, mu=self.alpha, x_star=self._x_star.copy(),
                          f_star=self._f_star)
