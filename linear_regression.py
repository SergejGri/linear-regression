import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotter


# Model Evaluation and Loss Function

# The primary goal of this project to get a full understanding of the regression logic.
# I will use the Mean Squared Error method as loss function J(w, b) ) = 1/n SUM_{0}^{n} (y_i - [m*x_i + t])**2
# y_i and x_i are the actual values. The idea is to "tweek" m and t to get best possible value for y. Each SUM-step delivers a difference bewteen the actual y and the predicted y.
# The squaring ensures the two effects. First, thelarger error values are penaltied more and get rid of negative values.
# t the y-axis intercept, m slope, alpha learning rate



def gradient_descent(x, y, m, t, alpha):
    N = len(x)
    y_pred = m * x + t

    dm = (-2/N) * np.sum((y - y_pred) * x)
    dt = (-2/N) * np.sum(y - y_pred)

    m = m - alpha*dm
    t = t - alpha*dt

    loss = np.mean((y - y_pred)**2)

    return m, t, loss


def train(x, y, m, t, alpha, epochs):
    MSE = np.zeros((epochs, 2))
    history = {"epoch": [], "m": [], "t": [], "loss": []}

    for e in range(epochs):
        m, t, loss = gradient_descent(x, y, m, t, alpha)
        MSE[e] = [e , loss]
        if e % 5 == 0:
            print(f"epoch: {e}, MSE: {loss}")
            history["epoch"].append(e)
            history["m"].append(m)
            history["t"].append(t)
            history["loss"].append(loss)
            plotter.dual_plot(x, y, history, final=False)

    plotter.dual_plot(x, y, history, final=True)
    return m, t, MSE, history


def main():
    url = "data/test_variable_scatter3.csv"
    df = pd.read_csv(url)
    epochs = 150
    alpha = 0.00001
    m = 0.0
    t = 0.0
    x = df["x"].values
    y = df["y"].values
    
    m, t, loss, history = train(x, y, m, t, alpha, epochs)

  
      

if __name__ == "__main__":
    main()