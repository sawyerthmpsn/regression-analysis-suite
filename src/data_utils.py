import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

def prep_for_multi_reg(df: pd.DataFrame):
    cols = df.columns
    y = df[cols[len(cols) - 1]]
    x = df.drop(columns = cols[[len(cols) - 1]], axis = 1) # Assumes all columns besides the final are parameters
    return y, x# Make a pyplot graph of the data and regression
def plot_regression(y, x, fit, type):

      # Initialize the scatter plot
    plt.scatter(x, y, s = 5)

    if type == 'linear':

        # Plot the linear fit
        b0, b1 = fit.params[0], fit.params[1]
        x_line = np.linspace(min(x), max(x), 100)
        y_line = b1 * x_line + b0
        plt.plot(x_line, y_line, color='red', label='Regression Line')

    elif type == 'logistic':

        # Generate x values for the plot
        x_values = np.linspace(x.min(), x.max(), 100)

        x_df = pd.DataFrame({'x': x_values})
        x_df = sm.add_constant(x_df)  # This adds the intercept column

        predicted_probabilities = fit.predict(x_df)

        # Create the plot
        plt.scatter(x, y, label='Data')
        plt.plot(x_values, predicted_probabilities, color='red', label='Logistic Regression')
        plt.xlabel('x')
        plt.ylabel('Probability of y=1')
        plt.title('Logistic Regression Example')
        plt.legend()
    
    plt.show()

def prep_for_simple_reg(df: pd.DataFrame, response, parameter):
    return np.array(df[response]), np.array(df[parameter])

