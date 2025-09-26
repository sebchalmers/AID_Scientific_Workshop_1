import os
import numpy as np
import matplotlib.pyplot as plt

# Output folder (same as this file)
BASE = os.path.dirname(os.path.abspath(__file__))
os.makedirs(BASE, exist_ok=True)

# ----- Synthetic return distribution (mixture with a heavier LEFT tail) -----
xs = np.linspace(-5, 5, 1600)
# Mixture: main near 1.0, small heavy left tail near -2.5
pdf = (
    0.78 * np.exp(-0.5 * ((xs - 1.0) / 1.0) ** 2) / (np.sqrt(2 * np.pi) * 1.0)
    + 0.22 * np.exp(-0.5 * ((xs + 2.5) / 0.7) ** 2) / (np.sqrt(2 * np.pi) * 0.7)
)
pdf /= np.trapz(pdf, xs)  # normalize

# CDF for quantiles / VaR
cdf = np.cumsum(pdf)
cdf /= cdf[-1]

# Risk parameter (lower tail of returns)
alpha = 0.10  # focus on worst 10% outcomes
var_idx = np.searchsorted(cdf, alpha)
VaR = xs[var_idx]

# Expected return
E = np.trapz(xs * pdf, xs)

# ----- Discrete atoms (C51-style) -----
# Support [vmin, vmax], evenly spaced atoms, weights ~ pdf(xs_atoms)
vmin, vmax, num_atoms = -5.0, 5.0, 51
atoms = np.linspace(vmin, vmax, num_atoms)
# Use pdf value at atoms and renormalize (just an illustrative projection)
# (Alternatively, sample and form a histogram—this is cleaner for a schematic.)
pdf_atoms = np.interp(atoms, xs, pdf)
pdf_atoms = np.maximum(pdf_atoms, 1e-12)
weights = pdf_atoms / np.sum(pdf_atoms)  # normalized

# ----- Plot -----
plt.close("all")
fig, ax = plt.subplots(figsize=(4.6, 2.9))





# Discrete atoms (stems): grey for most, red for those in the lower tail
for z, w in zip(atoms, weights):
    h = w * (vmax - vmin)  # scale just for visibility of stems
    if z <= VaR:
        ax.vlines(z, 0, h, linestyles=":", linewidth=2.2, color="red")
        ax.plot(z, h, "o", color="red", markersize=3.5)
    else:
        ax.vlines(z, 0, h, linestyles=":", linewidth=1.8, color="gray")
        ax.plot(z, h, "o", color="gray", markersize=3.0)

# VaR and Expected return lines
ax.axvline(VaR, linestyle="--", linewidth=1.2)
ax.text(VaR + 0.05, ax.get_ylim()[1] * 0.84, r"VaR$_{\alpha}$", fontsize=9)
ax.axvline(E, linestyle="-.", linewidth=1.2)
ax.text(E + 0.05, ax.get_ylim()[1] * 0.65, r"$\mathbb{E}[R]$", fontsize=9)

# Aesthetics: no ticks, tight layout
ax.set_xticks([])
ax.set_yticks([])
fig.tight_layout()
fig.savefig(os.path.join(BASE, "DistributionalRL.pdf"))
plt.close(fig)

print("Wrote figure:", os.path.join(BASE, "DistributionalRL.pdf"))