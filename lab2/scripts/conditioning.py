import numpy as np
import matplotlib.pyplot as plt

from oracles.media_mix import MediaMixOracle, ScaledOracle

def hessian_stats(H):
    """Минимальное и максимальное собственные значения и число обусловленности."""
    eig = np.linalg.eigvalsh(H)
    lam_min = eig.min()
    lam_max = eig.max()

    if lam_min <= 0:
        kappa_value = np.inf
    else:
        kappa_value = lam_max / lam_min

    return kappa_value, lam_min, lam_max, eig


def solution_point(oracle):
    """Находим точку оптимума."""
    if hasattr(oracle, "x_star"):
        return oracle.x_star.copy()
    return oracle._newton()


def print_table(title, headers, rows):
    """Печатает таблицу без дополнительных библиотек."""
    print(f"\n{title}")
    widths = [
        max(len(str(headers[i])),
            *(len(str(row[i])) for row in rows))
        for i in range(len(headers))
    ]

    print(" | ".join(str(v).ljust(widths[i])
                     for i, v in enumerate(headers)))
    print("-+-".join("-" * w for w in widths))

    for row in rows:
        print(" | ".join(str(v).ljust(widths[i])
                         for i, v in enumerate(row)))


oracle = MediaMixOracle()
x0 = np.zeros(oracle.dim) + 1e-3
x_star = solution_point(oracle)

points = [
    ("Стартовая точка x0", x0),
    ("Оптимум x*", x_star),
]

rows = []

for name, x in points:
    k, lo, hi, eig = hessian_stats(oracle.hess(x))
    rows.append((
        name,
        f"{lo:.6f}",
        f"{hi:.6f}",
        f"{k:.2f}",
    ))

print_table(
    "1. Обусловленность гессиана",
    ["Точка", "lambda_min", "lambda_max", "kappa"],
    rows,
)


spreads = [1, 5, 10, 30, 100]
kappas_b = []
rows = []

for spread in spreads:
    o = MediaMixOracle(b_spread=spread)
    xs = solution_point(o)
    k, lo, hi, _ = hessian_stats(o.hess(xs))

    kappas_b.append(k)
    rows.append((
        spread,
        f"{lo:.6f}",
        f"{hi:.6f}",
        f"{k:.2f}",
    ))

print_table(
    "2. Влияние b_spread на обусловленность в оптимуме",
    ["b_spread", "lambda_min", "lambda_max", "kappa"],
    rows,
)


mus = [1, 3, 10, 30, 100]
kappas_mu = []
overspend = []
rows = []

for mu in mus:
    o = MediaMixOracle(mu_pen=mu)
    xs = solution_point(o)
    k, lo, hi, _ = hessian_stats(o.hess(xs))

    excess = xs.sum() - o.B

    kappas_mu.append(k)
    overspend.append(excess)

    rows.append((
        mu,
        f"{lo:.6f}",
        f"{hi:.6f}",
        f"{k:.2f}",
        f"{excess:+.6f}",
    ))

print_table(
    "3. Влияние штрафа mu_pen",
    ["mu_pen", "lambda_min", "lambda_max",
     "kappa", "sum(x)-B"],
    rows,
)



alphas = [0.001, 0.01, 0.05, 0.2, 1.0]
kappas_a = []
rows = []

for alpha in alphas:
    o = MediaMixOracle(alpha=alpha)
    xs = solution_point(o)
    k, lo, hi, _ = hessian_stats(o.hess(xs))

    kappas_a.append(k)
    rows.append((
        alpha,
        f"{lo:.6f}",
        f"{hi:.6f}",
        f"{k:.2f}",
    ))

print_table(
    "4. Влияние регуляризации alpha",
    ["alpha", "lambda_min", "lambda_max", "kappa"],
    rows,
)

plt.figure(figsize=(6, 4))
plt.plot(alphas, kappas_a, "o-")
plt.xscale("log")
plt.xlabel("alpha")
plt.ylabel("Число обусловленности kappa")
plt.title("Влияние регуляризации на обусловленность")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("kappa_vs_alpha.png", dpi=150)
plt.close()



d = 1.0 / np.sqrt(oracle.hess_diag(x0))
scaled = ScaledOracle(oracle, d)
u_star = x_star / d
u0 = x0 / d

k_before, lo_before, hi_before, eig_x = hessian_stats(
    oracle.hess(x_star)
)
k_after, lo_after, hi_after, eig_u = hessian_stats(
    scaled.hess(u_star)
)

print_table(
    "5. Эффект масштабирования по Якоби",
    ["Показатель", "Исходная задача", "После масштабирования"],
    [
        ("lambda_min", f"{lo_before:.6f}", f"{lo_after:.6f}"),
        ("lambda_max", f"{hi_before:.6f}", f"{hi_after:.6f}"),
        ("kappa", f"{k_before:.2f}", f"{k_after:.2f}"),
    ],
)

print("\nСобственные значения гессиана в исходных координатах:")
print(np.round(eig_x, 4))

print("\nСобственные значения после масштабирования:")
print(np.round(eig_u, 4))

# Сравнение спектров
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
    "s-",
    label="После масштабирования",
)
plt.xlabel("Номер собственного значения")
plt.ylabel("Собственное значение (лог. шкала)")
plt.title("Спектр гессиана")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("spectrum.png", dpi=150)
plt.close()



def contour_on_plane(ax, f, ref, i, j, title):
    """Строит линии уровня, изменяя две координаты."""
    gi = np.linspace(0, max(2.5 * ref[i], 1e-6), 120)
    gj = np.linspace(0, max(2.5 * ref[j], 1e-6), 120)
    Z = np.zeros((len(gj), len(gi)))

    for a, vi in enumerate(gi):
        for b, vj in enumerate(gj):
            p = ref.copy()
            p[i] = vi
            p[j] = vj
            Z[b, a] = f(p)

    ax.contour(gi, gj, Z, levels=25)
    ax.plot(ref[i], ref[j], "r*", markersize=12)
    ax.set_xlabel(f"x{i + 1}")
    ax.set_ylabel(f"x{j + 1}")
    ax.set_title(title)


i, j = 0, oracle.dim - 1

fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))

contour_on_plane(
    ax[0], oracle.f, x_star, i, j,
    "Исходные координаты",
)
contour_on_plane(
    ax[1], scaled.f, u_star, i, j,
    "После масштабирования",
)

plt.tight_layout()
plt.savefig("contours.png", dpi=150)
plt.close()



fig, ax = plt.subplots(1, 2, figsize=(10, 4))

ax[0].plot(spreads, kappas_b, "o-")
ax[0].set_xscale("log")
ax[0].set_xlabel("b_spread")
ax[0].set_ylabel("kappa")
ax[0].set_title("Обусловленность от b_spread")
ax[0].grid(True, alpha=0.3)

ax[1].plot(mus, kappas_mu, "o-")
ax[1].set_xscale("log")
ax[1].set_xlabel("mu_pen")
ax[1].set_ylabel("kappa")
ax[1].set_title("Обусловленность от mu_pen")
ax[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("conditioning.png", dpi=150)
plt.close()


def newton_iters(o, start, tol=1e-6, max_iter=100):
    """Количество шагов Ньютона до достижения допуска."""
    x = start.copy()

    for k in range(max_iter):
        g = o.grad(x)

        if np.linalg.norm(g) <= tol:
            return k

        H = o.hess(x)

        try:
            step = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            return max_iter

        fx = o.f(x)
        t = 1.0

        while t > 1e-10 and o.f(x - t * step) > fx:
            t *= 0.5

        if t <= 1e-10:
            return max_iter

        x = x - t * step

    return max_iter


def gd_iters(o, start, tol=1e-6, max_iter=20000):
    """GD с постоянным шагом 1/L."""
    xs = solution_point(o)

    L_start = np.linalg.eigvalsh(o.hess(start)).max()
    L_star = np.linalg.eigvalsh(o.hess(xs)).max()
    L = max(L_start, L_star)

    if L <= 0:
        raise ValueError("Максимальное собственное значение должно быть положительным")

    x = start.copy()

    for k in range(max_iter):
        g = o.grad(x)

        if np.linalg.norm(g) <= tol:
            return k

        x = x - g / L

    return max_iter


rows = []

for name, o, start in [
    ("Исходная", oracle, x0),
    ("Масштабированная", scaled, u0),
]:
    gd = gd_iters(o, start)
    nt = newton_iters(o, start)

    rows.append((
        name,
        gd,
        "Да" if gd < 20000 else "Нет",
        nt,
        "Да" if nt < 100 else "Нет",
    ))

print_table(
    "6. Сравнение GD и Ньютона (допуск 1e-6)",
    ["Задача", "Итерации GD", "GD сошёлся?",
     "Итерации Ньютона", "Ньютон сошёлся?"],
    rows,
)


print("\nГрафики сохранены:")
for filename in [
    "kappa_vs_alpha.png",
    "spectrum.png",
    "contours.png",
    "conditioning.png",
]:
    print(f"  {filename}")
