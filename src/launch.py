import numpy as np
import pandas as pd

file_name = 'data/data.csv'


def main():
    data = pd.read_csv(file_name)
    print(data)

if __name__ == "__main__":
    main()