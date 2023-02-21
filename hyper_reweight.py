import matplotlib.pyplot as plt

import seaborn as sns
import numpy as np
plt.rcParams["font.family"] = "Times New Roman"

reweight=[0,0.3,0.7,1,2]
performance=[1.10,0.89,0.47,0.37,0.43]
std=[0.49,0.48,0.35,0.33,0.73]
fig, axs = plt.subplots(figsize=(5, 3))

plt.plot(reweight, performance, '-o',linestyle='dashed',linewidth=2)
plt.fill_between(reweight, np.array(performance) + np.array(std), np.array(performance) - np.array(std), alpha=0.2)
axs.set_xlabel("Proportion of the reweight", fontsize=10)
axs.set_ylabel("AMSE", fontsize=10)

axs.set_label(["DBRNet"])

# axs.set_title("AMSE across different proportions of reweight", fontsize=12)
# axs.set_ylim(-135, 135)
axs.set_ylim(-0.5, 2)
plt.tight_layout()
plt.savefig("reweight.pdf")