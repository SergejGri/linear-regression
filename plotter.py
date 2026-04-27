import os
import numpy as np
import imageio.v2 as imageio
import matplotlib.pyplot as plt


def dual_plot(x, y, history, final=False):
    m = history["m"][-1]
    t = history["t"][-1]
    epoch = history["epoch"][-1]

    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    axs[0].scatter(x, y, color="blue", alpha=0.5, label="Test data")
    x_line = np.array([min(x), max(x)])
    y_line = m * x_line + t
    axs[0].plot(x_line, y_line, color="red", linewidth=2)
    axs[0].set_title("Fit of test data")
    axs[0].set_xlabel("$x$")
    axs[0].set_ylabel("$y$")
    equation = rf"$y = {m:.2f}x + {t:.2f}$"

    axs[0].plot(x_line, y_line, color="red", linewidth=2, label=f"Fit: {equation}")
    axs[0].legend()

    axs[1].plot(history["epoch"], history["loss"], color="orange", marker="x")
    axs[1].set_title("Mean Square Error")
    axs[1].set_xlabel("Epoche")
    axs[1].set_ylabel(rf"$J(m, t)$")

    plt.tight_layout()
    filename = "plots/lin_regression.png" if final else f"plots/frame_{epoch}.png"
    plt.savefig(filename, dpi=300, facecolor="white")
    plt.close(fig)
    if final:
        create_gif()


def create_gif(folder="plots", output_name="training_progress.gif", fps=5):
    images = []
    
    # Get all files that start with "frame_"
    filenames = [f for f in os.listdir(folder) if f.startswith("frame_") and f.endswith(".png")]
    
    filenames.sort(key=lambda x: int(x.split("_")[1].split(".")[0]))

    print(f"Found {len(filenames)} frames. Creating GIF...")

    for filename in filenames:
        file_path = os.path.join(folder, filename)
        images.append(imageio.imread(file_path))

    imageio.mimsave("plots/"+ output_name, images, fps=fps, loop=0)
    print(f"Done! GIF saved as {output_name}")