import matplotlib.pyplot as plt
import numpy as np
import sys
import matplotlib
import math


cublas = np.zeros(64)


file = sys.argv[1]
fp = open(file)

lines = fp.readlines()

for i in range(64):
    cublas[i] = float(lines[13 + i * 14].split()[1])


x1 = ['256-256-256', '256-256-512', '256-256-1024', '256-256-2048', '256-512-256', '256-512-512', '256-512-1024', '256-512-2048', '256-1024-256', '256-1024-512', '256-1024-1024', '256-1024-2048',
      '256-2048-256', '256-2048-512', '256-2048-1024', '256-2048-2048']

x2 = ['512-256-256', '512-256-512', '512-256-1024', '512-256-2048', '512-512-256', '512-512-512', '512-512-1024', '512-512-2048', '512-1024-256', '512-1024-512', '512-1024-1024', '512-1024-2048',
      '512-2048-256', '512-2048-512', '512-2048-1024', '512-2048-2048']

x3 = ['1024-256-256', '1024-256-512', '1024-256-1024', '1024-256-2048', '1024-512-256', '1024-512-512', '1024-512-1024', '1024-512-2048', '1024-1024-256', '1024-1024-512', '1024-1024-1024', '1024-1024-2048',
      '1024-2048-256', '1024-2048-512', '1024-2048-1024', '1024-2048-2048']

x4 = ['2048-256-256', '2048-256-512', '2048-256-1024', '2048-256-2048', '2048-512-256', '2048-512-512', '2048-512-1024', '2048-512-2048', '2048-1024-256', '2048-1024-512', '2048-1024-1024', '2048-1024-2048',
      '2048-2048-256', '2048-2048-512', '2048-2048-1024', '2048-2048-2048']

fig, ax = plt.subplots()
ax.grid(True)
ax.set_xlabel("M-N-K")
ax.xaxis.set_ticks(np.arange(0, 16, 1))
ax.set_xticklabels(x1, rotation=45)
ax.set_xlim(-0.5, 15.5)
ax.set_ylabel("Performance (Gflop/s)")
plt.plot(cublas[:16], label="Performance", color="blue", marker='x')
plt.title("Performance in cublas implementation - 1")
plt.savefig("cublas1.png", bbox_inches="tight")

fig, ax = plt.subplots()
ax.grid(True)
ax.set_xlabel("M-N-K")
ax.xaxis.set_ticks(np.arange(0, 16, 1))
ax.set_xticklabels(x2, rotation=45)
ax.set_xlim(-0.5, 15.5)
ax.set_ylabel("Performance (Gflop/s)")
plt.plot(cublas[16:32], label="Performance", color="blue", marker='x')
plt.title("Performance in cublas implementation - 2")
plt.savefig("cublas2.png", bbox_inches="tight")

fig, ax = plt.subplots()
ax.grid(True)
ax.set_xlabel("M-N-K")
ax.xaxis.set_ticks(np.arange(0, 16, 1))
ax.set_xticklabels(x3, rotation=45)
ax.set_xlim(-0.5, 15.5)
ax.set_ylabel("Performance (Gflop/s)")
plt.plot(cublas[32:48], label="Performance", color="blue", marker='x')
plt.title("Performance in cublas implementation - 3")
plt.savefig("cublas3.png", bbox_inches="tight")

fig, ax = plt.subplots()
ax.grid(True)
ax.set_xlabel("M-N-K")
ax.xaxis.set_ticks(np.arange(0, 16, 1))
ax.set_xticklabels(x4, rotation=45)
ax.set_xlim(-0.5, 15.5)
ax.set_ylabel("Performance (Gflop/s)")
plt.plot(cublas[:48], label="Performance", color="blue", marker='x')
plt.title("Performance in cublas implementation - 4")
plt.savefig("cublas4.png", bbox_inches="tight")
