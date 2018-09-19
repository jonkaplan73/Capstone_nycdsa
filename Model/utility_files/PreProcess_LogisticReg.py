
import json
import pandas as pd
import re
import random
import numpy as np
from sklearn.model_selection import train_test_split
from collections import Counter
#from sklearn.metrics import roc_curve, auc, roc_auc_score
import sklearn.metrics # import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
import sklearn.model_selection as ms
import time


#function to check if the variable is defined and defines it to an empty string if not

def preprocess(df):
    ## Remove ".0" from MCC column
    df['mcc']=df['mcc'].apply(lambda x: str(x).strip(".0"))
    ## Change mcc nan to NaN
    df['mcc'] = np.where(df['mcc']=='nan', np.nan, df['mcc'])
    df['merchant_string'] = df['merchant_string'].apply(lambda x: x.lower())

## Map by MCC as well
mcc_dict = {'6011': 'atm', '6010': 'atm', '7523':'parking'}

def mcc_dict_funct(df, col_origin, col_output, mcc_dict):
    for key, value in mcc_dict.items():
            df[col_output] = np.where(df[col_origin]==key, value, df[col_output])

def most_common_words(df, col):
    # Turn merchant string into list and flatten list of sublists
    words_merchant_string_2 = [elem for sublist in df[col].tolist() for elem in sublist]

    # Get DataFrame of words with their count
    wordcnt_df = pd.DataFrame.from_dict(dict(Counter(words_merchant_string_2)), orient='index')\
        .reset_index().rename(columns={"index": "keyword", 0: "cnt"})\
        .sort_values(by='cnt',ascending=False)
    most_common_words = list(wordcnt_df['keyword'][0:1000])
    most_common_words1 = ["\\" + x if x[0] == "*" else x for x in most_common_words]
    return wordcnt_df, most_common_words, most_common_words1

def dummify_data(df, most_common_words1):
    for keyword in most_common_words1:
        df[keyword] = np.where(df['merchant_string'].str.contains(keyword),1,0)