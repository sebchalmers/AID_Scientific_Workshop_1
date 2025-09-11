import numpy as np
import matplotlib.pyplot as plt
import os

BASE = os.path.dirname(__file__)

# Sample distribution of returns (heavy-tailed example)
np.random.seed(1)
returns = np.random.standard_t(df=3, size=8000)  # heavy-tailed distribution

# Entropic risk functional
def entropic_risk(z_samples, lam):
    if lam == 0:
        return np.mean(z_samples)
    else:
        return -1.0/lam * np.log(np.mean(np.exp(-lam * z_samples)))

# Range of lambda values (risk sensitivity)
lams = np.linspace(-2, 2, 100)
rho_vals = [entropic_risk(returns, lam) for lam in lams]

# Expected value baseline
exp_val = np.mean(returns)

# Plot (match styling with HeavyTail_vs_Mean)
fig, ax = plt.subplots(figsize=(4, 2.5))

# Main curve and baseline
ax.plot(lams, rho_vals, color='tab:blue', linewidth=1.8)
ax.axhline(exp_val, color='black', linestyle='--', linewidth=1.2)

# Annotations (compact, fontsize=8)
ylim = ax.get_ylim()
ax.annotate('Risk-neutral (mean)',
            xy=(0, exp_val), xytext=(0.6, exp_val + 0.15*(ylim[1]-ylim[0])),
            arrowprops=dict(arrowstyle='->', lw=1.0), fontsize=8, ha='left', color='black')
ax.annotate('Risk-averse',
            xy=(1.4, rho_vals[-1]), xytext=(1.7, rho_vals[-1] - 0.25*(ylim[1]-ylim[0])),
            arrowprops=dict(arrowstyle='->', lw=1.0), fontsize=8, ha='left', color='black')
ax.annotate('Risk-seeking',
            xy=(-1.4, rho_vals[0]), xytext=(-3.3, rho_vals[0] + 0.2*(ylim[1]-ylim[0])),
            arrowprops=dict(arrowstyle='->', lw=1.0), fontsize=8, ha='left', color='black')

# Cosmetics: same axis labeling style, but hide ticks
ax.set_xlabel(r'Risk parameter $\lambda$')
ax.set_ylabel(r'Risk score $\rho_\lambda(Z)$')
ax.set_yticks([])
ax.set_xticks([])

fig.tight_layout()
outpath = os.path.join(BASE, 'RiskFunctional_Rho.pdf')
fig.savefig(outpath, format='pdf')
print(f"Saved figure to {outpath}")