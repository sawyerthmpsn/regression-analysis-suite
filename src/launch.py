import numpy as np
import pandas as pd
import linreg as lin
import matplotlib as plt

file_name = 'data/multiple_regression_dataset.csv'

def main():
    df = pd.read_csv(file_name)
    lin.summary_simple(df, 'Hours_Studied', 'Exam_Score', graph = True)

if __name__ == "__main__":
    main()
