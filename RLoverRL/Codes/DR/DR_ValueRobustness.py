import matplotlib.pyplot as plt
import numpy as np
import os

# Value under different distributions
x = np.linspace(0,1,100)
mean_val = 0.8 - 0.2*x   # optimistic
robust_val = 0.5*np.ones_like(x)  # worst-case
dr_val = 0.65 - 0.1*x   # DRMDP compromise

plt.figure(figsize=(3,2.2))
plt.plot(x, mean_val, 'b-', lw=2, label="Risk-neutral")
plt.plot(x, dr_val, 'g--', lw=2, label="Distributionally robust")
plt.plot(x, robust_val, 'r-', lw=2, label="Robust worst-case")

plt.xlabel("Uncertainty level"); plt.ylabel("Value")
plt.legend(fontsize=8)
plt.tight_layout()
plt.savefig("DR_ValueRobustness.pdf")