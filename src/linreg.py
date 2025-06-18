import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import scipy.stats as stats
import statsmodels.api as sm
matplotlib.style.use('ggplot')

def summary_multiple(df: pd.DataFrame):

    # Separate dependent and independent variables
    cols = df.columns
    y = df[cols[0]]
    x = df.drop(columns = cols[0], axis = 1) # Assumes all columns besides 0 are parameters

    # Create the linear regression model
    x = sm.add_constant(x)
    model = sm.OLS(y, x)
    fit = model.fit()

    # Linear regression summary output
    print(fit.summary())


# Simple linear regression: can pick one response variable and one parameter, and makes a visual plot
def summary_simple(df: pd.DataFrame, response, parameter):

    # Separate dependent and independent variables
    df = df[[parameter, response]].dropna()
    y = np.array(df[response])
    x = np.array(df[parameter])

    # Create the linear regression model
    x_temp = sm.add_constant(x)
    model = sm.OLS(y, x_temp)
    fit = model.fit()

    # Linear regression summary output
    print(fit.summary())

    # Initialize the scatter plot
    plt.scatter(x, y, s = 5)

    # Add the linear fit
    b0, b1 = fit.params[0], fit.params[1]
    x_line = np.linspace(min(x), max(x), 100)
    y_line = b1 * x_line + b0
    plt.plot(x_line, y_line, color='red', label='Regression Line')
    plt.show()