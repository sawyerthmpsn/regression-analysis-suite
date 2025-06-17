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
    y = (df[cols[0]])
    x = df.drop(columns = cols[0], axis = 1) # Assumes all columns besides 0 are parameters
    print(x)
    print(y)

    # Create the linear regression model
    x = sm.add_constant(x)
    model = sm.OLS(y, x)
    fit = model.fit()

    # Summarize data and plot results
    print(fit.summary())


# Simple linear regression: can pick one response variable and one parameter, and makes a visual plot
def summary_simple(df: pd.DataFrame, response, paramteter):


    ## Initialize the scatter plot
#    plt.scatter(x, y, s = len(df))

    ## Add the linear fit
#    b0, b1 = fit.params[0], fit.params[1]
#    x_line = np.linspace(min(x), max(x), 100)
#    y_line = b1 * x_line + b0
#    plt.plot(x_line, y_line, color='red', label='Regression Line')
#    plt.show()
    pass