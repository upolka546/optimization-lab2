import numpy as np
import matplotlib.pyplot as plt
from oracles.media_mix import MediaMixOracle, ScaledOracle


def kappa(H):
    """Число обусловленности симметричной матрицы H."""
    e = np.linalg.eigvalsh(H)          # собственные значения по возрастанию
    return e.max() / e.min(), e.min(), e.max()


def solution_point(oracle):
    """Точка оптимума x*. Если у оракула нет x_star, берём внутренний метод Ньютона."""
    if hasattr(oracle, "x_star"):
        return oracle.x_star
    return oracle._newton()


# κ в стартовой точке и в оптимуме
oracle = MediaMixOracle()                       # параметры по умолчанию
x0 = np.zeros(oracle.dim) + 1e-3                # стартовая точка
x_star = solution_point(oracle)

for name, x in [("x0 (старт)", x0), ("x* (оптимум)", x_star)]:
    k, lo, hi = kappa(oracle.hess(x))
    print(f"{name}: kappa={k:.1f}, lambda_min={lo:.3f}, lambda_max={hi:.3f}")

# как κ зависит от разброса каналов b_spread 
spreads = [1, 5, 10, 30, 100]
kappas_b = []
for s in spreads:
    o = MediaMixOracle(b_spread=s)
    kappas_b.append(kappa(o.hess(solution_point(o)))[0])
    print(f"b_spread={s}: kappa={kappas_b[-1]:.1f}")

# как κ зависит от штрафа бюджета mu_pen
mus = [1, 3, 10, 30, 100]
kappas_mu, overspend = [], []
for m in mus:
    o = MediaMixOracle(mu_pen=m)
    xs = solution_point(o)
    kappas_mu.append(kappa(o.hess(xs))[0])
    overspend.append(xs.sum() - o.B)            # перерасход бюджета
    print(f"mu_pen={m}: kappa={kappas_mu[-1]:.1f}, sum(x)-B={overspend[-1]:+.3f}")

# κ после предобуславливания (масштабирование по Якоби) 
d = 1.0 / np.sqrt(oracle.hess_diag(x0))         # d_i = 1/sqrt(H_ii)
scaled = ScaledOracle(oracle, d)
u_star = x_star / d                             # та же точка в новых координатах
k_before = kappa(oracle.hess(x_star))[0]
k_after = kappa(scaled.hess(u_star))[0]
print(f"kappa до: {k_before:.1f}, после масштабирования: {k_after:.1f}")

# Графики 
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].plot(spreads, kappas_b, "o-")
ax[0].set(xscale="log", xlabel="b_spread", ylabel="kappa", title="κ от разброса каналов")
ax[1].plot(mus, kappas_mu, "o-")
ax[1].set(xscale="log", xlabel="mu_pen", ylabel="kappa", title="κ от штрафа бюджета")
plt.tight_layout()
plt.savefig("conditioning.png", dpi=150)
print("Saved: conditioning.png")
