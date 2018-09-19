
import pandas as pd
import numpy as np

from sklearn import naive_bayes
from nltk.tokenize import word_tokenize
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from nltk import bigrams

import matplotlib.pyplot as plt
import matplotlib as matplotlib
import os

import time
import pickle as pickle
import sys as sys


hasLabel,noLabel,X_train,X_test, y_train, y_test = '','','','','',''

def NB(cleanMerchantCol = 'merchant_string_clean', writeResults = True, probThresholdAnalysis = True,
       makeAllPredictions = False,originalMerchantCol = 'merchant_string',
       hasLabel=hasLabel,noLabel=noLabel,X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test):

    count_vec = CountVectorizer(ngram_range=(1, 4), token_pattern= r'\b\w+\b', min_df=1)
    X_train_count = count_vec.fit_transform(X_train[cleanMerchantCol])
    multiNB = MultinomialNB()
    cntvecMNB = multiNB.fit(X_train_count, y_train)


    def save_as_pickled_object(obj, filepath):
        max_bytes = 2**31 - 1
        bytes_out = pickle.dumps(obj)
        n_bytes = sys.getsizeof(bytes_out)
        with open(filepath, 'wb') as f_out:
            for idx in range(0, n_bytes, max_bytes):
                f_out.write(bytes_out[idx:idx+max_bytes])

    save_as_pickled_object(count_vec,'final_NB_countvec.sav')
    save_as_pickled_object(multiNB,'final_multiNB_model.sav')


  

