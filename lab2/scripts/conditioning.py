
import numpy as np
import matplotlib.pyplot as plt
from oracles.media_mix import MediaMixOracle


def kappa(H):
    e = np.linalg.eigvalsh(H)
    return e.max() / e.min(), e.min(), e.max()


def solution_point(oracle):
    return oracle.info.x_star.copy()


class ScaledOracle:
    def __init__(self, oracle, d):
        self.oracle = oracle
        self.d = np.asarray(d, dtype=float)
        self.dim = oracle.dim
        self.name = f"scaled({oracle.name})"

    def f(self, u):
        return self.oracle.f(self.d * u)

    def grad(self, u):
        return self.d * self.oracle.grad(self.d * u)

    def hess(self, u):
        H = self.oracle.hess(self.d * u)
        return self.d[:, None] * H * self.d[None, :]

    def hess_diag(self, u):
        return self.d**2 * self.oracle.hess_diag(self.d * u)

    def hess_vec(self, u, v):
        return self.d * self.oracle.hess_vec(self.d * u, self.d * v)


oracle = MediaMixOracle()
x0 = np.zeros(oracle.dim) + 1e-3
x_star = solution_point(oracle)

for name, x in [("x0 (старт)", x0), ("x* (оптимум)", x_star)]:
    k, lo, hi = kappa(oracle.hess(x))
    print(f"{name}: kappa={k:.1f}, lambda_min={lo:.3f}, lambda_max={hi:.3f}")

rhos = [0.1, 0.3, 1.0, 3.0, 10.0]
kappas_rho, overspend = [], []

for rho in rhos:
    o = MediaMixOracle(rho=rho)
    xs = solution_point(o)
    kappas_rho.append(kappa(o.hess(xs))[0])
    overspend.append(xs.sum() - o.B)
    print(f"rho={rho}: kappa={kappas_rho[-1]:.1f}, sum(x)-B={overspend[-1]:+.3f}")

lams = [0.001, 0.01, 0.05, 0.2, 1.0]
kappas_lam = []

for lam in lams:
    o = MediaMixOracle(lam=lam)
    xs = solution_point(o)
    kappas_lam.append(kappa(o.hess(xs))[0])
    print(f"lam={lam}: kappa={kappas_lam[-1]:.1f}")

gammas = [1.0, 3.0, 10.0, 30.0, 100.0]
kappas_gamma = []

for gamma in gammas:
    o = MediaMixOracle(gamma=gamma)
    xs = solution_point(o)
    kappas_gamma.append(kappa(o.hess(xs))[0])
    print(f"gamma={gamma}: kappa={kappas_gamma[-1]:.1f}")

d = 1.0 / np.sqrt(oracle.hess_diag(x0))
scaled = ScaledOracle(oracle, d)
u_star = x_star / d

k_before = kappa(oracle.hess(x_star))[0]
k_after = kappa(scaled.hess(u_star))[0]
print(f"kappa до: {k_before:.1f}, после масштабирования: {k_after:.1f}")

plt.figure(figsize=(5, 4))
plt.plot(lams, kappas_lam, "o-")
plt.xscale("log")
plt.yscale("log")
plt.xlabel("lam")
plt.ylabel("kappa")
plt.title("κ от регуляризации")
plt.tight_layout()
plt.savefig("kappa_vs_lam.png", dpi=150)
plt.close()


def contour_on_plane(ax, f, ref, i, j, title):
    gi = np.linspace(0, max(1e-6, 2.5 * ref[i]), 100)
    gj = np.linspace(0, max(1e-6, 2.5 * ref[j]), 100)
    Z = np.zeros((len(gj), len(gi)))

    for a, vi in enumerate(gi):
        for b, vj in enumerate(gj):
            p = ref.copy()
            p[i], p[j] = vi, vj
            Z[b, a] = f(p)

    ax.contour(gi, gj, Z, levels=25)
    ax.plot(ref[i], ref[j], "r*", markersize=12)
    ax.set(title=title, xlabel=f"x{i+1}", ylabel=f"x{j+1}")


i, j = 0, oracle.dim - 1
fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
contour_on_plane(ax[0], oracle.f, x_star, i, j, "Исходные координаты")
contour_on_plane(ax[1], scaled.f, u_star, i, j, "После масштабирования")
plt.tight_layout()
plt.savefig("contours.png", dpi=150)
plt.close()

fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].plot(rhos, kappas_rho, "o-")
ax[0].set(xscale="log", yscale="log", xlabel="rho", ylabel="kappa", title="κ от штрафа бюджета")
ax[1].plot(gammas, kappas_gamma, "o-")
ax[1].set(xscale="log", yscale="log", xlabel="gamma", ylabel="kappa", title="κ от штрафа отрицательных координат")
plt.tight_layout()
plt.savefig("conditioning.png", dpi=150)
plt.close()

print("Saved: kappa_vs_lam.png, contours.png, conditioning.png")
