import matplotlib.pyplot as plt
import numpy as np
import os

# make folder

# Samples
np.random.seed(0)
data = np.random.normal(0,1,100)

plt.figure(figsize=(3,3))
plt.hist(data, bins=20, density=True, alpha=0.6, label=r"Empirical $\hat{P}$")
x = np.linspace(-3,3,200)
plt.plot(x, 1/np.sqrt(2*np.pi)*np.exp(-x**2/2), 'r-', lw=2, label="True P?")

# Plot alternative plausible transition distributions
alternative_params = [(-0.5, 1.2), (0.5, 0.8), (0, 1.5)]
for mu, sigma in alternative_params:
    plt.plot(x, 1/(sigma*np.sqrt(2*np.pi))*np.exp(-(x-mu)**2/(2*sigma**2)), '--', lw=1.5, label=r"$P \in \mathcal{P}$")

plt.legend(fontsize=8)
plt.xlabel("x"); plt.ylabel("density")
plt.tight_layout()
plt.savefig("DR_AmbiguitySet.pdf")