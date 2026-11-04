import numpy as np
import matplotlib.pyplot as plt

from oracles.media_mix import MediaMixOracle

def solution_point(oracle):
    """Возвращает оптимум медиамикса."""
    return oracle.info.x_star.copy()


def hessian_stats(H):
    """Собственные значения и число обусловленности."""
    eig = np.linalg.eigvalsh(H)

    lam_min = eig.min()
    lam_max = eig.max()

    kappa = lam_max / lam_min if lam_min > 0 else np.inf

    return kappa, lam_min, lam_max, eig


def print_table(title, headers, rows):
    """Печатает таблицу."""

    print("\n" + title)

    widths = [
        max(len(str(headers[i])),
            *(len(str(row[i])) for row in rows))
        for i in range(len(headers))
    ]

    print(" | ".join(
        str(v).ljust(widths[i])
        for i, v in enumerate(headers)
    ))

    print("-+-".join("-" * w for w in widths))

    for row in rows:
        print(" | ".join(
            str(v).ljust(widths[i])
            for i, v in enumerate(row)
        ))


oracle = MediaMixOracle()

x0 = np.full(oracle.dim, oracle.B / oracle.dim)
x_star = solution_point(oracle)



rows = []

for name, x in [
    ("x0 = 0", x0),
    ("x* (оптимум)", x_star),
]:

    kappa, lam_min, lam_max, eig = hessian_stats(
        oracle.hess(x)
    )

    rows.append((
        name,
        f"{kappa:.2f}",
        f"{lam_min:.3f}",
        f"{lam_max:.2f}",
    ))


print_table(
    "1. Обусловленность гессиана",
    ["Точка", "kappa", "lambda_min", "lambda_max"],
    rows,
)


# Масштабируем координаты по диагонали гессиана
d = 1.0 / np.sqrt(oracle.hess_diag(x_star))


# Новый оракул в координатах x = d * u
class ScaledOracle:

    def __init__(self, oracle, d):
        self.oracle = oracle
        self.d = d
        self.dim = oracle.dim

    def f(self, u):
        return self.oracle.f(self.d * u)

    def grad(self, u):
        return self.d * self.oracle.grad(self.d * u)

    def hess(self, u):
        H = self.oracle.hess(self.d * u)

        return self.d[:, None] * H * self.d[None, :]


scaled = ScaledOracle(oracle, d)

u_star = x_star / d


# Сравниваем собственные значения
k_before, lo_before, hi_before, eig_x = hessian_stats(
    oracle.hess(x_star)
)

k_after, lo_after, hi_after, eig_u = hessian_stats(
    scaled.hess(u_star)
)


print_table(
    "2. Эффект масштабирования по Якоби",
    ["Показатель", "До", "После"],
    [
        ("lambda_min", f"{lo_before:.4f}", f"{lo_after:.4f}"),
        ("lambda_max", f"{hi_before:.4f}", f"{hi_after:.4f}"),
        ("kappa", f"{k_before:.2f}", f"{k_after:.2f}"),
    ],
)


# График спектра
plt.figure(figsize=(7, 4))

plt.semilogy(
    range(1, len(eig_x) + 1),
    eig_x,
    "o-",
    label="Исходная задача",
)

plt.semilogy(
    range(1, len(eig_u) + 1),
    eig_u,
    "s--",
    label="После масштабирования",
)

plt.xlabel("Номер собственного значения")
plt.ylabel("Собственное значение")
plt.title("Спектр гессиана")

plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig("spectrum.png", dpi=150)
plt.show()


spreads = [1, 5, 10, 30, 100]

kappas_b = []
rows = []

for spread in spreads:

    # Создаём отдельную задачу
    o = MediaMixOracle()

    # Изменяем параметр насыщения каналов
    o.b = o.b * spread

    # Пересчитываем оптимум для новой задачи
    o._x_star, o._f_star = o._solve()

    xs = solution_point(o)

    kappa, lam_min, lam_max, eig = hessian_stats(
        o.hess(xs)
    )

    kappas_b.append(kappa)

    rows.append((
        spread,
        f"{lam_min:.4f}",
        f"{lam_max:.4f}",
        f"{kappa:.2f}",
    ))


print_table(
    "3. Влияние b_spread на обусловленность",
    ["b_spread", "lambda_min", "lambda_max", "kappa"],
    rows,
)


# График зависимости обусловленности
plt.figure(figsize=(6, 4))

plt.plot(spreads, kappas_b, "o-")

plt.xscale("log")
plt.xlabel("b_spread")
plt.ylabel("Число обусловленности kappa")

plt.title("Влияние b_spread")
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig("conditioning_b_spread.png", dpi=150)
plt.show()
