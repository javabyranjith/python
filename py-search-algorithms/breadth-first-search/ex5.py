# EXPERIMENT - 5

# Implementation of locally weighted regression algorithm
# pip install matplotlib

from math import ceil, pi
import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt

def lowess(x, y, f, iterations):
    n = len(x)
    r = int(ceil(f * n))
    
    # Compute neighborhood window h[i]
    h = [np.sort(np.abs(x - x[i]))[r] for i in range(n)]
    
    # Compute weights
    w = np.clip(np.abs((x[:, None] - x[None, :]) / h), 0.0, 1.0)
    w = (1 - w ** 3) ** 3

    yest = np.zeros(n)
    delta = np.ones(n)

    for iteration in range(iterations):
        for i in range(n):
            weights = delta * w[:, i]
            b = np.array([np.sum(weights * y), np.sum(weights * y * x)])
            A = np.array([
                [np.sum(weights), np.sum(weights * x)],
                [np.sum(weights * x), np.sum(weights * x * x)]
            ])
            beta = linalg.solve(A, b)
            yest[i] = beta[0] + beta[1] * x[i]

        residuals = y - yest
        s = np.median(np.abs(residuals))
        delta = np.clip(residuals / (6.0 * s), -1, 1)
        delta = (1 - delta ** 2) ** 2

    return yest

# Generate noisy sine wave data
n = 100
x = np.linspace(0, 2 * pi, n)
y = np.sin(x) + 0.3 * np.random.randn(n)

# Apply LOWESS smoothing
f = 0.25
iterations = 3
yest = lowess(x, y, f, iterations)

# Plot the results
plt.figure(figsize=(10, 5))
plt.plot(x, y, "r.", label="Noisy Data")
plt.plot(x, yest, "b-", label="LOWESS Smoothed")
plt.legend()
plt.title("LOWESS Smoothing")
plt.xlabel("x")
plt.ylabel("y")
plt.show()
