import numpy as np
import pandas as pd
import linreg as lin
import logreg as log
import matplotlib as plt

file_name = 'data/multiple_regression_dataset.csv'

def main():
    df = pd.read_csv(file_name)
    lin.summary_multiple(df)

if __name__ == "__main__":
    main()
