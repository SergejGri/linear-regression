import matplotlib.pyplot as plt
import numpy as np

def plot(x, y, m, t, loss):
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    axs[0].plot(loss[:, 0], loss[:, 1], color='orange', marker="x")
    axs[0].set_title("Mean Square Error")
    axs[0].set_xlabel("Epoche")

    axs[1].scatter(x, y, color='blue', alpha=0.5, label='Data')
    x_line = np.array([min(x), max(x)])
    y_line = m * x_line + t
    axs[1].plot(x_line, y_line, color='red', linewidth=2, label='Regression')
    axs[1].set_title('Regression')
    axs[1].set_xlabel('x')
    equation = rf"$y = {m:.2f}x + {t:.2f}$"

    axs[1].plot(x_line, y_line, color='red', linewidth=2, label=f"Fit: {equation}")
    axs[1].legend()

    plt.tight_layout()
    plt.savefig('plots/lin_regression.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()