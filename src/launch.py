import numpy as np
import pandas as pd
import linreg as lin
import logreg as log
import data_utils

file_name = 'data/Students Social Media Addiction.csv'

def main():
    df = pd.read_csv(file_name)
    df = data_utils.convert_data_types(df, file_name)
    lin.summary_multiple(df)

if __name__ == "__main__":
    main()
