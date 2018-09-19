
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
    X_test_count = count_vec.transform(X_test[cleanMerchantCol])
    predicted_brand_test = cntvecMNB.predict(X_test_count)

    print('The score of multinomial naive bayes is on Train: %.4f' %multiNB.score(X_train_count, y_train))
    print('The score of multinomial naive bayes is on Test: %.4f' %multiNB.score(X_test_count, y_test))

    score = multiNB.score(X_test_count, y_test)

    probability = pd.DataFrame(multiNB.predict_proba(X_test_count))
    probs = list(probability.max(axis=1))


    results = {'Merchant_String':list(X_test[cleanMerchantCol]),'Labels':list(y_test),'Predictions':list(predicted_brand_test),'Confidence':list(probs)}
    results = pd.DataFrame(results)
    results['CorrectPrediction']=np.where(results['Predictions']==results['Labels'],1,0)

    resultsbyBrand=results.groupby('Labels').agg(['mean','count'])['CorrectPrediction']
    resultsbyBrand=resultsbyBrand.sort_values(by='count', ascending=False)

    # save the model to disk
    def save_as_pickled_object(obj, filepath):
        max_bytes = 2**31 - 1
        bytes_out = pickle.dumps(obj)
        n_bytes = sys.getsizeof(bytes_out)
        with open(filepath, 'wb') as f_out:
            for idx in range(0, n_bytes, max_bytes):
                f_out.write(bytes_out[idx:idx+max_bytes])

    save_as_pickled_object(cntvecMNB,'finalized_NB_model.sav')
    save_as_pickled_object(count_vec,'final_NB_countvec.sav')
    save_as_pickled_object(multiNB,'final_multiNB_model.sav')

    if writeResults == True:
        try:
            os.mkdir("Results_"+str(score))
        except:
            pass

        #write model
        save_as_pickled_object(cntvecMNB,'./Results_'+str(score)+'/finalized_NB_model.sav')
        #Write Results by Observation-------------------------------------------------------------------------------------------------------------------------
        results.to_csv('./Results_'+str(score)+'/ResultsByPrediction_'+ str(score ) +'.csv',index=True)


        #Write Results by Brand-------------------------------------------------------------------------------------------------------------------------
        resultsbyBrand.to_csv('./Results_'+str(score)+'/ResultsByBrand_'+ str(score ) +'.csv',index=True)

        #Write Threshold Analysis-------------------------------------------------------------------------------------------------------------------------
        if probThresholdAnalysis == True:
            pctPredictAll = []
            percentPredicted = []
            pctPedictPrecise = []
            for i in np.linspace(0, .99, num=100):
                pctPredictAll.append(np.mean(results['CorrectPrediction']))
                percentPredicted.append(len(results[results['Confidence']>=i])/len(results))
                pctPedictPrecise.append(np.mean(results[results['Confidence']>=i]['CorrectPrediction']))
            threshold = pd.DataFrame({'threshold':list(np.linspace(0, .99, num=100)), 'Percent_Predicted':percentPredicted,'Percent_Pedicted_Over_Threshold':pctPedictPrecise,'Percent_Predicted_All_Observations':pctPredictAll})
            threshold.to_csv('./Results_'+str(score )+'/thresholdAnalysis.csv',index=False)

            plt.figure(figsize=(12,8))
            plt.plot(threshold.threshold, threshold.Percent_Predicted, color='g')
            plt.plot(threshold.threshold, threshold.Percent_Pedicted_Over_Threshold,color = 'blue')
            plt.plot(threshold.threshold, threshold.Percent_Predicted_All_Observations,color = 'orange')
            matplotlib.rcParams.update({'font.size': 8})
            plt.xlabel('Prediction Threshold')
            plt.ylabel('Predcition Metrics')
            matplotlib.rcParams.update({'font.size': 15})
            plt.title('Threshold vs Accuracy')
            matplotlib.rcParams.update({'font.size': 15})
            plt.legend()
            plt.savefig('./Results_'+str(score )+'/Threshold_Analysis_Graph.png')
            print(plt.show())
       


        if makeAllPredictions == True:
            X_noLabel_count = count_vec.transform(noLabel[cleanMerchantCol])
            predicted_brand_noLabel = cntvecMNB.predict(X_noLabel_count)
            probability = pd.DataFrame(multiNB.predict_proba(X_noLabel_count))
            probs_noLabel = list(probability.max(axis=1))
            results_noLabel = {'ID':list(noLabel.index.values),'mapped_brand':list(noLabel['mapped_brand']),'Merchant_String_Original':list(noLabel[originalMerchantCol]),'Merchant_String_Clean':list(noLabel['merchant_string_clean']),'Predictions':list(predicted_brand_noLabel),'Confidence':list(probs_noLabel)}
            results_noLabel = pd.DataFrame(results_noLabel)

            results_noLabel.to_csv('./Results_'+str(score)+'/ResultsByPrediction_NoLabel.csv',index=False)

            return(results_noLabel)


def NB_help():
    print("""def NB(cleanMerchantCol = 'merchant_string_clean', writeResults = True, probThresholdAnalysis = True,
       makeAllPredictions = False,originalMerchantCol = 'merchant_string',
       hasLabel=hasLabel,noLabel=noLabel,X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test):""")

