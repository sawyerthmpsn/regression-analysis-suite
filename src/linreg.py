import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import scipy.stats as stats
import statsmodels.api as sm
import data_utils
matplotlib.style.use('ggplot')

# Use OLS method to make a line of best fit
def fit_OLS(y, x):
    x_temp = sm.add_constant(x)
    model = sm.OLS(y, x_temp)
    return model.fit()

# Make a pyplot graph of the data and regression
def plot_regression(y, x, fit):

      # Initialize the scatter plot
    plt.scatter(x, y, s = 5)

    # Add the linear fit
    b0, b1 = fit.params[0], fit.params[1]
    x_line = np.linspace(min(x), max(x), 100)
    y_line = b1 * x_line + b0
    plt.plot(x_line, y_line, color='red', label='Regression Line')
    plt.show()

def summary_multiple(df: pd.DataFrame):

    # Separate dependent and independent variables
    y, x = data_utils.prep_for_multi_reg(df)

    # Create the linear regression model
    fit = fit_OLS(y, x)

    # Linear regression summary output
    print(fit.summary())


# Simple linear regression: can pick one response variable and one parameter, and makes a visual plot
def summary_simple(df: pd.DataFrame, response, parameter, graph):

    # Separate dependent and independent variables
    y, x = data_utils.prep_for_simple_reg(df, response, parameter)

    # Create the linear regression model
    fit = fit_OLS(y, x)

    # Linear regression summary output
    print(fit.summary())

    if graph:
        plot_regression(y, x, fit)