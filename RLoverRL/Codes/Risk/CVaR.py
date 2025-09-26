import os
import numpy as np
import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.abspath(__file__))
os.makedirs(BASE, exist_ok=True)

# ================= Figure 1: RS_CVaR_Saddle.pdf =================
# Mixture distribution; show VaR_alpha, a generic eta, and CVaR tail mean
np.random.seed(0)
xs = np.linspace(-4, 4, 800)
pdf = 0.7*np.exp(-0.5*((xs-0.0)/1.0)**2)/np.sqrt(2*np.pi) \
    + 0.3*np.exp(-0.5*((xs-1.8)/0.8)**2)/(0.8*np.sqrt(2*np.pi))
pdf /= np.trapz(pdf, xs)

alpha = 0.9
cdf = np.cumsum(pdf); cdf /= cdf[-1]
var_idx = np.searchsorted(cdf, alpha)
VaR = xs[var_idx]
mask_tail = xs >= VaR

# Tail mean (CVaR approx under pdf)
tail_mass = np.trapz(pdf[mask_tail], xs[mask_tail])
CVaR = np.trapz(xs[mask_tail]*pdf[mask_tail], xs[mask_tail]) / max(tail_mass, 1e-12)

# Pick a generic eta line near VaR to illustrate min over eta
eta = VaR + 0.15

fig, ax = plt.subplots(figsize=(4.4, 2.8))
ax.plot(xs, pdf, linewidth=1.3)
ax.fill_between(xs[mask_tail], 0, pdf[mask_tail], alpha=0.25)
ax.axvline(VaR, linestyle='--', linewidth=1.0)
ax.text(VaR+0.05, ax.get_ylim()[1]*0.82, r'VaR$_{\alpha}$', fontsize=9)
ax.axvline(eta, linestyle=':', linewidth=1.5)
ax.text(eta+0.05, ax.get_ylim()[1]*0.64, r'$\eta$', fontsize=9)
ax.axvline(CVaR, linestyle='-', linewidth=1.5)
ax.text(CVaR+0.05, ax.get_ylim()[1]*0.50, r'CVaR$_{\alpha}$ (tail mean)', fontsize=9)
ax.set_xticks([]); ax.set_yticks([])
fig.tight_layout()
fig.savefig(os.path.join(BASE, 'RS_CVaR_Saddle.pdf'))
plt.close(fig)
print('Wrote figures to', BASE)