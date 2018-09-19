import pandas as pd
import numpy as np
import sklearn.model_selection as ms
import random


def stratfiedTrainTestSplit_help():
    print("""def stratfiedTrainTestSplit(df, brandCol ='mapped_brand', holdoutPct = .2, ones = 'sample'):
        
        the input 'ones' can be 'train' 'test' or 'sample' and will determine where to put the ones
        
        the function outputs a series of data frames in order
        hasLabel,noLabel,X_train, X_test, y_train, y_test""")



def stratfiedTrainTestSplit(df, brandCol ='mapped_brand', holdoutPct = .2, ones = 'sample'):

    cols = list(df.columns)
    cols.remove(brandCol)
    col = cols[0]

    groupedBrand = df.groupby([brandCol]).agg(['count'])
    ListOfOnes = groupedBrand[list((groupedBrand[col]<2)['count'])].index
    ListOfOnes = ListOfOnes.tolist()


    df = df.fillna('None')

    hasLabel = df[df[brandCol]!='None']
    noLabel = df[df[brandCol]=='None']

    #all ones go to the train set
    train1s = hasLabel[hasLabel[brandCol].isin(ListOfOnes)]
    y_1s = train1s[brandCol]
    x_1s = train1s.drop(brandCol,axis=1)

    xy = hasLabel[~hasLabel[brandCol].isin(ListOfOnes)]
    y = xy[brandCol]
    x = xy.drop(brandCol,axis=1)

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(x, y,random_state = 42,
                                                        stratify=y, 
                                                        test_size=holdoutPct) 

    y_test[y_test=='None'][brandCol] = None

    if ones == 'train':
        X_train = pd.concat([X_train,x_1s])
        y_train = pd.concat([y_train,y_1s])
    if ones == 'test':
        X_test = pd.concat([X_test,x_1s])
        y_test = pd.concat([y_test,y_1s])
    if ones == 'sample':
        random.seed(42)
        index = np.random.choice(len(y_1s),round(len(y_1s)*holdoutPct), replace=False)
        X_train = pd.concat([X_train,x_1s.iloc[index]])
        y_train = pd.concat([y_train,y_1s.iloc[index]])
        X_test = pd.concat([X_test,x_1s.iloc[~index]])
        y_test = pd.concat([y_test,y_1s.iloc[~index]])

    return hasLabel,noLabel,X_train, X_test, y_train, y_test




    