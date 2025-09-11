import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['text.usetex'] = True
FS = 40

# Define underlying mixture distribution for sampling
Mu  = [-0.3, 0.2]
Std = [0.1, 0.2]
Wei = [0.2, 0.8]


# ===================== Few-data case: show many plausible fits =====================
# Small sample size
Ns_small = 40
Nbins_small = 40

# Resample a tiny dataset from the same mixture
S_small = []
for k in range(Ns_small):
    bin_idx = np.random.choice(range(len(Mu)), p=Wei)
    S_small.append(np.random.normal(Mu[bin_idx], Std[bin_idx]))
S_small = np.array(S_small)

# Histogram for small data
Hf, sf = np.histogram(S_small, bins=Nbins_small, range=(-1,1), density=False)
sf = 0.5*sf[:-1] + 0.5*sf[1:]
Hf = Hf * Nbins_small / 2 / Ns_small

# Define grid for evaluating distributions
x = np.linspace(-1, 1, 1000)
# A single plausible fit from small data (e.g., Gaussian MLE)
mu_hat = float(np.mean(S_small))
sigma_hat = float(np.std(S_small) + 1e-6)
alt0 = np.exp(-(x-mu_hat)**2/(2*sigma_hat**2)) / (np.sqrt(2*np.pi)*sigma_hat)

# Additional plausible fits (mean/variance perturbed) - expanded to 8
alt_plausible = [
    (mu_hat - 0.20, max(0.12, 0.8*sigma_hat)),
    (mu_hat + 0.20, max(0.15, 1.2*sigma_hat)),
    (mu_hat - 0.05, max(0.25, 1.5*sigma_hat)),
    (mu_hat + 0.10, max(0.18, 1.1*sigma_hat)),
    (mu_hat - 0.30, max(0.10, 0.9*sigma_hat)),
    (mu_hat + 0.25, max(0.20, 1.3*sigma_hat)),
    (mu_hat - 0.15, max(0.22, 1.4*sigma_hat)),
    (mu_hat + 0.05, max(0.16, 1.0*sigma_hat)),
]

# Generate 8 sequential plots where each adds another plausible fit
for i in range(8):
    plt.close('all')
    fig, ax = plt.subplots(figsize=(12,6))

    # Histogram of few data points
    ax.fill_between(sf, Hf, np.zeros(len(sf)), color='b', step='mid', label=r"Empirical $\hat{P}$ (few)")
    # Also show raw samples as dots on baseline to emphasize scarcity
    ax.plot(S_small, np.zeros_like(S_small), 'k.', markersize=1, alpha=0.7)

    # Removed plotting of alt0 (red plausible fit)

    # Add i alternative plausible fits (red dotted) and track max_alt
    max_alt = 0.0
    for j in range(i):
        mu_p, sig_p = alt_plausible[j]
        alt = np.exp(-(x-mu_p)**2/(2*sig_p**2)) / (np.sqrt(2*np.pi)*sig_p)
        ax.plot(x, alt, 'r:', linewidth=3, label=r"$P \in \mathcal{P}$" if j == 0 else None)
        max_alt = max(max_alt, float(np.max(alt)))
    ymax = 1.5*max(float(np.max(Hf)), max_alt)
    ax.set_ylim([0, ymax])

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel(r"$\mathbf{s}_{+}$", fontsize=FS)
    ax.set_ylabel(r"$\mathbf{P}(s_+|s,a)$", fontsize=FS, color='r')

    if i == 0:
        ax.legend(fontsize=18, loc='upper left')

    fname = f"DistViewFew_{i}.pdf"
    fig.savefig(fname, dpi='figure', bbox_inches='tight')
    print(f"Saved {fname}")