import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import scipy.stats as stats
import statsmodels.api as sm
import data_utils
matplotlib.style.use('ggplot')

def fit_MLE(y, x):
    x_temp = sm.add_constant(x)
    model = sm.Logit(y, x_temp)
    return model.fit()

def summary_multiple(df: pd.DataFrame):

    # Separate dependent and independent variables
    y, x = data_utils.prep_for_multi_reg(df)

    # Create the logistic regression model
    fit = fit_MLE(y, x)

    print (fit.summary())

def summary_simple(df: pd.DataFrame, response, parameter, graph):

    # Separate dependent and independent variables
    y, x = data_utils.prep_for_simple_reg(df, response, parameter)

    # Create the linear regression model
    fit = fit_MLE(y, x)

    # Linear regression summary output
    print(fit.summary())

    if graph:
        data_utils.plot_regression(y, x, fit, 'logistic')

