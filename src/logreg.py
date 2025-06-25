import numpy as np
import pandas as pd
from sklearn import feature_selection, linear_model

import data_utils

# feature selection
def feature_selector(y, x):

    # Uses Forward Sequential Selection to get the 5 most relevant parameters
    estimator = linear_model.LogisticRegression()
    selector = feature_selection.SequentialFeatureSelector(estimator = estimator,
                                                           n_features_to_select = 3,
                                                           direction = 'forward',
                                                           )
    return selector


# Use MLE method to make a line of best fit
def fit_MLE(y, x, x_selected, selector):

    # Fit the logistic regression model using the selected features
    model = linear_model.LogisticRegression()
    return model.fit(x_selected, y), selector.get_feature_names_out(x.columns)


def summary_multiple(df: pd.DataFrame):

    # Separate dependent and independent variables
    y, x = data_utils.prep_for_multi_reg(df)

    # Create the logistic regression model
    selector = feature_selector(y, x)
    x_selected = selector.fit_transform(x, y)
    fit, params = fit_MLE(y, x, x_selected, selector)

    # Logistic regression summary output
    results = [np.insert(params, 0, 'Intercept'), np.insert(fit.coef_, 0, fit.intercept_)]
    print(pd.DataFrame(results).to_string(header = False, index = False))