import matplotlib.pyplot as plt
import numpy as np
import sys
import matplotlib
import math

perf = np.zeros(4)

perf[0] = 112.651352

perf[1] = 186.383629

perf[2] = 365.505475

perf[3] = 2223.486822

barWidth = 0.1
# Set position of bar on X axis
r = np.arange(4)

fig, ax = plt.subplots()
ax.set_xlabel("Method")

ax.xaxis.set_ticks(np.arange(0, len(r), 1))
ax.set_xticklabels(['Naive', 'Coalesced', 'Reduced', 'cuBLAS'], rotation=45)
ax.set_xlim(-0.5, len(r) - 0.5)
#ax.set_ylim(0.0, 3.0)
ax.set_ylabel("Performance (Gflops)")

plt.bar(r, perf,
        width=barWidth, color='blue')


plt.title("Performance in M=N=K=2048")
plt.savefig("bar_performance.png", bbox_inches="tight")
