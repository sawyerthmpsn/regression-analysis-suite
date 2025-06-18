import numpy as np
import pandas as pd

def prep_for_multi_reg(df: pd.DataFrame):
    cols = df.columns
    y = df[cols[0]]
    x = df.drop(columns = cols[0], axis = 1) # Assumes all columns besides 0 are parameters
    return y, x

def prep_for_simple_reg(df: pd.DataFrame, response, parameter):
    return np.array(df[response]), np.array(df[parameter])