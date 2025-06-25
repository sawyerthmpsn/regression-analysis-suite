import numpy as np
import pandas as pd


def prep_for_multi_reg(df: pd.DataFrame):
    cols = df.columns
    y = df[cols[len(cols) - 1]]
    x = df.drop(columns = cols[[len(cols) - 1]], axis = 1) # Assumes all columns besides the final are parameters
    return y, x# Make a pyplot graph of the data and regression


def prep_for_simple_reg(df: pd.DataFrame, response, parameter):
    return np.array(df[response]), np.array(df[parameter])


# Edit this method if needed to suit the data being analyzed
def convert_data_types(df: pd.DataFrame, file_name: str):

    #--------------------Custom Preprocessing Procedures--------------------#

    if (file_name == 'data/Students Social Media Addiction.csv'):

        # Ignore Student ID
        df = df.drop('Student_ID', axis = 1)
    
        # Filter out all but entries with select values of Most_Used_Platform column
        platforms_blacklist = 'KakaoTalk', 'LINE', 'LinkedIn', 'VKontakte', 'WeChat', 'WhatsApp'
        for x in platforms_blacklist:
            df = df[df['Most_Used_Platform'] != x]

        # Sort countries into two categories: "USA" and "Other"
        df.loc[df['Country'] != 'USA', ['Country']] = 'Other'


    elif (file_name == 'data/synthetic_logistic_regression_dataset'):
        pass

    #-----------------------------------------------------------------------#

    # Get response variable column
    column_to_move = df.columns[-1]

    # One Hot Encoding for all categorical variables
    df = pd.get_dummies(df, drop_first=True)

    # Turn all encoded variables from type boolean to type int
    boolean_columns = df.select_dtypes(include='bool').columns
    df[boolean_columns] = df[boolean_columns].astype(int)

    # Move response variable to the last column
    other_columns = [col for col in df.columns if col != column_to_move]
    new_column_order = other_columns + [column_to_move]

    return df[new_column_order]