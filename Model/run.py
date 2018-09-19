import time
start_time = time.time()

import sys
sys.path.insert(0, './utility_files')
import cleanData


sys.path.insert(0, './utility_files')
import PreProcess_LogisticReg

import json
import pandas as pd
import os
import re
import numpy as np
import pickle


#Import data
dataFinal = pd.read_csv('./data/input_data.csv')
dataFinal['merchant_string'] = dataFinal['merchant_string'].astype(str)
dataFinal_copy = dataFinal.copy()

model = 'both' #'lr','nb', 'both'

if model !=  'lr':
    #Import nb model if they don't exist but of they are stored in memory then do not re-import them
    try:
        count_vec
        multiNB 
    except:
        print('model needs to import')
        def try_to_load_as_pickled_object_or_None(filepath):
            """
            This is a defensive way to write pickle.load, allowing for very large files on all platforms
            """
            max_bytes = 2**31 - 1

            input_size = os.path.getsize(filepath)
            bytes_in = bytearray(0)
            with open(filepath, 'rb') as f_in:
                for _ in range(0, input_size, max_bytes):
                    bytes_in += f_in.read(max_bytes)
            obj = pickle.loads(bytes_in)

            return obj

        count_vec = try_to_load_as_pickled_object_or_None('./NB_Models_ModelCreation/final_NB_countvec.sav')
        multiNB = try_to_load_as_pickled_object_or_None('./NB_Models_ModelCreation/final_multiNB_model.sav')

    remove_string = '[]' #'^[0-9]*[0-9]$|^www$|^com$|^ave|^street$|^st$|^road$|^and$|^inc$|^at$|^drive$|^of$|^main$|^the$|^[ewns]$|^#|^[0-9]*th$|^[0-9]*rd$|^1st$|^store$|^south$'
    split_string = "[- ./*']"
    cleanData.clean(dataFinal,col='clean_nb_merchant_string',rejoin_col_strings_bi=True,lowercase=True,split_string=split_string,remove_string = remove_string,join_mcc_bi=True)

    #put data into sparce matrix
    X_test_count = count_vec.transform(dataFinal['clean_nb_merchant_string'])
    predicted_brand_test = multiNB.predict(X_test_count)
    probability = pd.DataFrame(multiNB.predict_proba(X_test_count))
    probs = list(probability.max(axis=1))

    #add data back to the dataframe and save .csv file with mapped_brands
    dataFinal['Confidence_Level_NaiveBayes'] = probs
    dataFinal['Predict_NaiveBayes'] = list(predicted_brand_test)
    threshold =  .9997
    #dataFinal['Predict_NaiveBayes'] = np.where(dataFinal['Confidence_Level_NaiveBayes']>threshold,dataFinal['All_Predict_NaiveBayes'],'None')
    dataFinal.drop('clean_nb_merchant_string',axis=1,inplace=True)


if model !=  'nb':
    #Import nb model if they don't exist but of they are stored in memory then do not re-import them
    try:
        multiLog
    except:
        print('model needs to import')
        def try_to_load_as_pickled_object_or_None(filepath):
            """
            This is a defensive way to write pickle.load, allowing for very large files on all platforms
            """
            max_bytes = 2**31 - 1

            input_size = os.path.getsize(filepath)
            bytes_in = bytearray(0)
            with open(filepath, 'rb') as f_in:
                for _ in range(0, input_size, max_bytes):
                    bytes_in += f_in.read(max_bytes)
            obj = pickle.loads(bytes_in)

            return obj
    
    multiLog = try_to_load_as_pickled_object_or_None('./Logistic_Regression_ModelCreation/final_ML_model.sav')
    
    mcc_network_cols = list(pd.read_csv('./utility_files/mcc_network_cols.csv'))    
    mostCommonWords = list(pd.read_csv('./utility_files/most_common_words1.csv'))

    PreProcess_LogisticReg.preprocess(dataFinal_copy)


    remove_string = '^[0-9]*[0-9]$|^www$|^com$|^ave|^street$|^road$|^and$|^inc$|^at$|^drive$|^of$|^main$|^the$|^[ewns]$|^#|^[0-9]*th$|^3rd$|^2nd$|^1st$|^store$|^st$|^rd$|^blvd$|^hwy$|^dr$'
    split_string = '[-./ ]'

    cleanData.clean(dataFinal_copy, old_col='merchant_string', col='clean_lr_merchant_string',split_string=split_string,
        remove_string = remove_string,lowercase=False, remove_empty_strings_bi=True,
        join_mcc_bi=False,rejoin_col_strings_bi=False)

    #wordcnt_df, most_common_words, most_common_words1 = PreProcess_LogisticReg.most_common_words(dataFinal,'merchant_string_clean_lr')
    data_dummified = dataFinal_copy.copy()
   

    # Dummify data (on most common words in merchant string cleaned)
    PreProcess_LogisticReg.dummify_data(data_dummified, mostCommonWords)


    # Dummify additional columns (mcc, network) and drop merchant string column
    data_dummified=pd.get_dummies(data_dummified, prefix=['mcc', 'network'], columns=['mcc', 'network'])
    
    #data_dummified.columns.contain(del_cols), axis = 1)
    data_dummified = data_dummified.drop(['clean_lr_merchant_string','merchant_string'], axis = 1)

    new_cols = list(set(mcc_network_cols) - set(data_dummified.columns))
    for i in new_cols:
        data_dummified[i] = 0
    
    #put data into sparce matrix
    print(multiLog.predict(data_dummified))
    predicted_brand_lr = multiLog.predict(data_dummified)
    dataFinal['Predictions_Logistic_Regression'] = predicted_brand_lr

if model =='both':
    dataFinal['Final_mapped_brand_prediction'] =  np.where(dataFinal['Confidence_Level_NaiveBayes']>threshold,dataFinal['All_Predict_NaiveBayes'],dataFinal['Predictions_Logistic_Regression'])


print(dataFinal.head())
open('./data/output_data_labeled.csv', 'w').close()
dataFinal.to_csv('./data/output_data_labeled.csv',index=False)

print("--- %s seconds ---" % (time.time() - start_time))
 

