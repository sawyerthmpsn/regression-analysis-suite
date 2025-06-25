import numpy as np
import pandas as pd
from sklearn import feature_selection, linear_model
from sklearn.model_selection import cross_val_score, KFold

import data_utils


# feature selection
def feature_selector(y, x):
    
    # Uses Forward Sequential Selection to get the 5 most relevant parameters (optimal = 7)
    estimator = linear_model.LinearRegression()
    selector = feature_selection.SequentialFeatureSelector(estimator = estimator,
                                                           n_features_to_select = 7,
                                                           direction = 'forward',
                                                           )
    return selector


# Use OLS method to make a line of best fit
def fit_OLS(y, x, x_selected, selector):

    # Fit the linear regression model using the selected features
    model = linear_model.LinearRegression()
    return model.fit(x_selected, y), selector.get_feature_names_out(x.columns)


def cross_validate(model, x, y):

    # Perform a k-folds split, with about 20 data points per fold, to determine R2 values
    kf = KFold(n_splits=len(y)//20, shuffle=False) # shuffle for random splits, random_state for reproducibility
    scores = cross_val_score(model, x, y, cv = kf, scoring = 'r2')

    # Convert R2 scores to Adj R2
    n = len(y)
    p = len(x[0])
    for i in range(len(scores)):
        scores[i] = 1 - (1 - scores[i]) * (n - 1) / (n - p - 1)

    return scores


# Makes a linear regression model and prints the coefficients of the intercept and parameters
def summary_multiple(df: pd.DataFrame):

    # Separate dependent and independent variables
    y, x = data_utils.prep_for_multi_reg(df)

    # Create the linear regression model
    selector = feature_selector(y, x)
    x_selected = selector.fit_transform(x, y)
    fit, params = fit_OLS(y, x, x_selected, selector)
    scores = cross_validate(fit, x_selected, y)

    # Linear regression summary output
    results = [np.insert(params, 0, 'Intercept'), np.insert(fit.coef_, 0, fit.intercept_)]
    print(pd.DataFrame(results).to_string(header = False, index = False))

    print(f'Mean R2: {np.mean(scores)}')
    print(f'SD of R2: {np.std(scores)}')