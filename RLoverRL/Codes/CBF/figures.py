import os
BASE = os.path.dirname(os.path.abspath(__file__))
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import PathPatch

theta = np.linspace(0, 2*np.pi, 400)
c = np.array([0.5, 0.5])     # obstacle center
r_obs = 0.25                 # obstacle radius (unsafe)
margin = 0.08
r_bar = r_obs + margin       # barrier boundary (h(x)=0)
target = np.array([1.1, 1.1])
t_r = 0.12                   # target radius (visual)

xlim = (-0.2, 1.4)
ylim = (-0.2, 1.4)


def _rect_path(xmin, xmax, ymin, ymax):
    return Path(
        [(xmin, ymin), (xmin, ymax), (xmax, ymax), (xmax, ymin), (xmin, ymin)],
        [Path.MOVETO, Path.LINETO, Path.LINETO, Path.LINETO, Path.CLOSEPOLY],
    )


def _circle_path(center, radius, clockwise=False, n=256):
    ang = np.linspace(0, 2*np.pi, n, endpoint=True)
    if clockwise:
        ang = ang[::-1]
    verts = [(center[0] + radius*np.cos(a), center[1] + radius*np.sin(a)) for a in ang]
    verts.append(verts[0])
    codes = [Path.MOVETO] + [Path.LINETO]*(len(verts)-2) + [Path.CLOSEPOLY]
    return Path(verts, codes)


def _add_safe_set_patch(ax):
    # Safe set C = outside barrier circle but inside the plotting window.
    # Build compound path: CCW rectangle (outer), CW circle (hole).
    R = _rect_path(xlim[0], xlim[1], ylim[0], ylim[1])
    C = _circle_path(c, r_bar, clockwise=True)
    comp = Path.make_compound_path(R, C)
    patch = PathPatch(comp, facecolor=(0.9, 0.9, 0.9), edgecolor='none', hatch='..', lw=0)
    ax.add_patch(patch)
    # Label the safe set
    ax.text(xlim[0] + 0.05, ylim[1] - 0.1, r"$\mathcal{C}$", fontsize=9, color="black")

def plot_scene(ax, show_safe_set=False):
    if show_safe_set:
        _add_safe_set_patch(ax)
    # Unsafe obstacle (gray disk)
    x_obs = c[0] + r_obs*np.cos(theta)
    y_obs = c[1] + r_obs*np.sin(theta)
    ax.fill(x_obs, y_obs, alpha=0.2)
    # Barrier boundary h(x)=0 (green and thicker)
    x_b = c[0] + r_bar*np.cos(theta)
    y_b = c[1] + r_bar*np.sin(theta)
    ax.plot(x_b, y_b, linestyle='--', color='green', linewidth=2.0)
    # Target region
    x_t = target[0] + t_r*np.cos(theta)
    y_t = target[1] + t_r*np.sin(theta)
    ax.plot(x_t, y_t, linewidth=1.2)
    # Formatting
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect('equal', 'box')
    ax.set_xticks([])
    ax.set_yticks([])

def h_and_grad(x):
    d = x - c
    n = np.linalg.norm(d) + 1e-9
    return n - r_bar, d / n  # h(x), grad h(x)

# -------- Figure 1: Unsafe trajectory --------
np.random.seed(7)
start = np.array([-0.1, -0.1])
x = start.copy()
traj_unsafe = [x.copy()]
for _ in range(240):
    v = target - x
    v = v / (np.linalg.norm(v) + 1e-9) * 0.035
    n = np.random.randn(2)*0.005
    x = x + v + n
    traj_unsafe.append(x.copy())
traj_unsafe = np.array(traj_unsafe)

fig, ax = plt.subplots(figsize=(3,3))
plot_scene(ax, show_safe_set=True)
ax.plot(traj_unsafe[:,0], traj_unsafe[:,1], linewidth=1.1)
fig.tight_layout()
plt.savefig(os.path.join(BASE, "CBF_Traj_unsafe.eps"), format="eps")
plt.close(fig)

# -------- Figure 2: QP projection (toy half-space) --------
fig, ax = plt.subplots(figsize=(4,3))
u1_b = 0.4
ax.plot([u1_b, u1_b], [-0.2, 1.2], linewidth=1.2)
ax.fill_betweenx(
    y=[-0.2, 1.2],
    x1=u1_b,
    x2=1.4,
    facecolor=(0.9, 0.9, 0.9),
    hatch='..',
    alpha=0.5
)  # feasible (safe) side consistent with safe set shading

u_des = np.array([0.15, 0.9])
u_safe = np.array([u1_b, u_des[1]])  # projection
ax.plot(*u_des, 'o'); ax.text(u_des[0]-0.12, u_des[1]+0.06, r"$u_{\mathrm{des}}$")
ax.plot(*u_safe, 'o'); ax.text(u_safe[0]+0.02, u_safe[1]+0.06, r"$u_{\mathrm{safe}}$")
ax.annotate("", xy=u_safe, xytext=u_des, arrowprops=dict(arrowstyle="->", linewidth=1.0))

ax.set_xlim(-0.1, 1.4); ax.set_ylim(-0.2, 1.2)
ax.set_xlabel(r"$u_1$"); ax.set_ylabel(r"$u_2$")
ax.set_aspect('equal', 'box'); ax.set_xticks([]); ax.set_yticks([])
fig.tight_layout()
plt.savefig(os.path.join(BASE, "CBF_QP_projection.eps"), format="eps")
plt.close(fig)

# -------- Figure 3: Safe trajectory via CBF projection each step --------
alpha = 2.0  # class-K gain
np.random.seed(7)
x = start.copy()
traj_safe = [x.copy()]
for _ in range(240):
    u_des = target - x
    u_des = u_des / (np.linalg.norm(u_des) + 1e-9) * 0.035
    n = np.random.randn(2)*0.005
    u_nom = u_des + n
    h, grad = h_and_grad(x)
    cbf_val = grad.dot(u_nom) + alpha*h
    if cbf_val < 0.0:
        # Minimal correction along grad h to satisfy cbf_val >= 0
        u_nom = u_nom + (-cbf_val) * grad
    x = x + u_nom
    traj_safe.append(x.copy())
traj_safe = np.array(traj_safe)

fig, ax = plt.subplots(figsize=(3,3))
plot_scene(ax, show_safe_set=True)
ax.plot(traj_unsafe[:,0], traj_unsafe[:,1], linestyle=':', linewidth=0.9)  # original unsafe
ax.plot(traj_safe[:,0], traj_safe[:,1], linewidth=1.2)                     # corrected safe
fig.tight_layout()
plt.savefig(os.path.join(BASE, "CBF_Traj_safe.eps"), format="eps")
plt.close(fig)

print("Saved: CBF_Traj_unsafe.eps, CBF_QP_projection.eps, CBF_Traj_safe.eps in", BASE)