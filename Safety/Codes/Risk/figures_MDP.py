import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.stats import gaussian_kde

# Output folder
BASE = os.path.dirname(__file__)

# ---------------- Setup: two return distributions ----------------
np.random.seed(0)
# Light-tailed (Gaussian-ish) and heavy-tailed (Student-t) returns
light_tail = np.random.normal(loc=0.0, scale=0.8, size=8000)
heavy_tail = np.random.standard_t(df=2, size=8000)  # heavier tails (rare large losses)

# Risk level (worst alpha tail)
alpha = 0.10  # focus on worst 10% outcomes (left tail)

# ---------------- Helper: lower-tail quantile (generic tail focus) ----------------

def lower_tail_quantile(samples, alpha):
    s = np.sort(samples)
    n = len(s)
    k = max(1, int(np.floor(alpha * n)))
    return s[k-1]  # α-quantile (lower tail)

var_H = lower_tail_quantile(heavy_tail, alpha)
mean_L, mean_H = light_tail.mean(), heavy_tail.mean()

# ---------------- KDE for visualization ----------------
x_vals = np.linspace(-8, 8, 1200)
kde_light = gaussian_kde(light_tail)
kde_heavy = gaussian_kde(heavy_tail)
y_light = kde_light(x_vals)
y_heavy = kde_heavy(x_vals)

# Tail mask for shading (left of lower-tail quantile of heavy-tail)
mask_tail_H = x_vals <= var_H

# ---------------- Plot ----------------
fig, ax = plt.subplots(figsize=(5.2, 3.0))

# Densities
ax.plot(x_vals, y_light, color='tab:blue', linewidth=1.8, label='Light-tailed')
ax.plot(x_vals, y_heavy, color='tab:red', linewidth=1.8, label='Heavy-tailed')

# Shade worst-α tail for the heavy-tailed distribution to emphasize risk focus
ax.fill_between(x_vals[mask_tail_H], 0, y_heavy[mask_tail_H], color='tab:red', alpha=0.20, label=f'Worst {int(alpha*100)}% tail (penalized)')

# Risk-neutral: means (thin dashed)
ax.axvline(mean_L, color='tab:blue', linestyle='--', linewidth=1.2)
ax.axvline(mean_H, color='tab:red', linestyle='--', linewidth=1.2)
# ax.text(mean_L, ax.get_ylim()[1]*0.92, 'mean', color='tab:blue', fontsize=8, ha='center', va='top')
# ax.text(mean_H, ax.get_ylim()[1]*0.92, 'mean', color='tab:red', fontsize=8, ha='center', va='top')

# Explanatory annotations
ax.annotate('Risk-neutral objective\n(maximize mean)',
            xy=(mean_H, ax.get_ylim()[1]*0.9), xytext=(2.7, ax.get_ylim()[1]*0.85),
            arrowprops=dict(arrowstyle='->', lw=1.0), fontsize=8, ha='left', color='black')

ax.annotate('Risk-sensitive objective:\nmaximize a risk-aware score $\\rho(R)$\nthat penalizes bad tails',
            xy=(x_vals[mask_tail_H].mean(), y_heavy[mask_tail_H].max()*0.6),
            xytext=(-6.8, ax.get_ylim()[1]*0.55),
            arrowprops=dict(arrowstyle='->', lw=1.0), fontsize=8, ha='left', color='black')

# Cosmetics
ax.set_xlabel('Return')
ax.set_ylabel('Density')
ax.set_yticks([])
ax.set_xticks([])
fig.tight_layout()

outpath = os.path.join(BASE, 'HeavyTail_vs_Mean.pdf')
fig.savefig(outpath, format='pdf')
print(f"Saved figure to {outpath}")