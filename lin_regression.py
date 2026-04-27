import numpy as np
from numpy.typing import NDArray
import pandas as pd
import plotter


def gradient_descent(
    x: NDArray,
    y: NDArray,
    slope: float,
    intercept: float,
    learning_rate: float
    ) -> tuple[float, float, float]:
    """
    Perform a single update step for linear regression using gradient descent.

    Args:
        x: Input features (independent variable).
        y: Target values (dependent variable).
        slope: The current weight/slope of the model.
        intercept: The current bias/intercept of the model.
        learning_rate: The step size for the update.

    Returns:
        A tuple containing updated slope, intercept and current loss.
    """
    N = len(x)
    y_pred = slope * x + intercept

    error = y - y_pred
    gradient_slope = (-2/N) * np.sum(error * x)
    gradient_intercept = (-2/N) * np.sum(error)

    new_slope = slope - learning_rate * gradient_slope
    new_intercept = intercept - learning_rate * gradient_intercept

    loss = np.mean(error**2)

    return new_slope, new_intercept, loss


def train_lin_regression(
    x: NDArray,
    y: NDArray,
    initial_slope: float,
    initial_intercept: float,
    learning_rate: float,
    epochs: int
    ) -> tuple[float, float, NDArray, dict[str, list[float]]]:

    """
    Train a linear regression model using gradient descent.

    Args:
        x: Input features.
        y: Target values.
        initial_slope: Starting weight.
        initial_intercept: Starting bias.
        learning_rate: Step size for updates.
        num_epochs: Number of training iterations.
        logging_interval: How often to print progress and plot.

    Returns:
        The final slope, intercept, loss history array, and a detailed history dictionary.
    """
    current_slope = initial_slope
    current_intercept = initial_intercept

    loss_history = np.zeros((epochs, 2))

    history = {
        "epoch": [],
        "slope": [],
        "intercept": [],
        "loss": []
        }

    for epoch in range(epochs):
        current_slope,
        current_intercept,
        current_loss = gradient_descent(x,
                                        y,
                                        slope = current_slope,
                                        intercept = current_intercept,
                                        learning_rate = learning_rate)
        
        loss_history[epoch] = [epoch , current_loss]
        
        if epoch % 5 == 0:
            print(f"epoch: {epoch}, current loss: {current_loss}")
            history["epoch"].append(epoch)
            history["slope"].append(current_slope)
            history["intercept"].append(current_intercept)
            history["loss"].append(current_loss)
            plotter.dual_plot(x, y, history, final=False)

    final_slope = current_slope
    final_intercept = current_intercept

    plotter.dual_plot(x, y, history, final=True)

    return final_slope, final_intercept, loss_history, history


def main():
    url = "data/test_variable_scatter3.csv"
    df = pd.read_csv(url)
    
    slope, intercept, loss, history = train_lin_regression(x = df["x"].values,
                                                           y = df["y"].values,
                                                           slope = 0.0,
                                                           intercept = 0.0,
                                                           learning_rate = 0.00001,
                                                           epochs = 150)


if __name__ == "__main__":
    main()