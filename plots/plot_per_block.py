import matplotlib.pyplot as plt
import numpy as np
import sys
import matplotlib
import math

x = ['4', '8', '16', '32']

naive = np.zeros(4)
coal = np.zeros(4)
red = np.zeros(4)

for file in sys.argv[1:]:
    fp = open(file)
    file_parts = file.split('_')
    size = int(file_parts[1][1:])
    lines = fp.readlines()
    naive[int(np.log2(size)) - 2] = float(lines[13].split()[1])
    coal[int(np.log2(size)) - 2] = float(lines[27].split()[1])
    red[int(np.log2(size)) - 2] = float(lines[41].split()[1])


fig, ax = plt.subplots()
ax.grid(True)
ax.set_xlabel("block size - tile size")
ax.xaxis.set_ticks(np.arange(0, 4, 1))
ax.set_xticklabels(['4', '8', '16', '32'], rotation=45)
ax.set_xlim(-0.5, 3.5)
ax.set_ylabel("Performance (Gflop/s)")
plt.plot(naive, label="Performance", color="blue", marker='x')
plt.title("Performance in naive implementation for M=N=K=2048")
plt.savefig("perf_naive.png", bbox_inches="tight")


fig, ax = plt.subplots()
ax.grid(True)
ax.set_xlabel("block size - tile size")
ax.xaxis.set_ticks(np.arange(0, 4, 1))
ax.set_xticklabels(['4', '8', '16', '32'], rotation=45)
ax.set_xlim(-0.5, 3.5)
ax.set_ylabel("Performance (Gflop/s)")
plt.plot(coal, label="Performance", color="blue", marker='x')
plt.title("Performance in coalesced implementation for M=N=K=2048")
plt.savefig("perf_coal.png", bbox_inches="tight")


fig, ax = plt.subplots()
ax.grid(True)
ax.set_xlabel("block size - tile size")
ax.xaxis.set_ticks(np.arange(0, 4, 1))
ax.set_xticklabels(['4', '8', '16', '32'], rotation=45)
ax.set_xlim(-0.5, 3.5)
ax.set_ylabel("Performance (Gflop/s)")
plt.plot(red, label="Performance", color="blue", marker='x')
plt.title("Performance in reduced implementation for M=N=K=2048")
plt.savefig("perf_red.png", bbox_inches="tight")
