

import pandas as pd
import re
import numpy as np

split_string = "[- /]"
remove_string = '[]'
data = ''
#function to check if the variable is defined and defines it to an empty string if not

def CID(variable):
  try:
    variable
  except:
    variable = '[]'
  return variable

def new_merchant_col(df, col_old, col_new):
  df[col_new] = df[col_old]

## Make a column lowercase
def lowercase_col(df, col):
  df[col]=df[col].apply(lambda x: x.lower())
    
## Split a column's string based on regex features
def split(df, col, split_string=split_string):
  df[col] = df[col].apply(lambda x: re.split(split_string, x))

## Remove strings that using regex
def remove_regex(df, col, remove_string=remove_string):
  # todo make regex_string easier to read
  df[col] = df[col].apply(lambda x: ['' if re.compile(remove_string).search(elem) is not None else elem for elem in x])

## Remove empty strings in a column's list ('' remaining from remove_regex)
#SM: don't understand purpose of this
def remove_empty_strings(df, col):
  df[col] = df[col].apply(lambda x: [elem for elem in x if elem])

def join_mcc(df,colBrand,colMcc):
  
  df[colBrand] = df[colBrand].apply(lambda x: ' '.join(x))

  combined_index = list(df[~np.isnan(df[colMcc])].index)
  combined_values = list(df[~np.isnan(df[colMcc])][colBrand] +' z' +\
  df[~np.isnan(df[colMcc])][colMcc].map(str).apply(lambda x: x.strip(".0") + 'z'))
  
  combinedNA_index = list(df[np.isnan(df[colMcc])].index)
  combinedNA_values = list(df[np.isnan(df[colMcc])][colBrand])

  dataNotNa = {'index':combined_index,'string':combined_values}            
  dataNA = {'index':combinedNA_index,'string':combinedNA_values}

  temp = pd.concat([pd.DataFrame(dataNotNa),pd.DataFrame(dataNA)], axis=0).sort_values(by='index').reset_index(drop=True).drop('index',axis=1)
  df[colBrand] = list(temp['string'])
  
  df[colBrand] = df[colBrand].apply(lambda x: re.split(' ', x))


## Other Functions
# Change mcc to string
def col_to_str(df, col):
  df[col] = df[col].apply(lambda x: str(x))    
    
# Rejoin strings in a column
def rejoin_col_strings(df, col):
  df[col] = df[col].apply(lambda x: ' '.join(x))

 
## Combine cleaning functions
def clean(df=data, old_col='merchant_string', col='merchant_string_clean',split_string=split_string, 
  remove_string = remove_string,lowercase=True, remove_empty_strings_bi=True, 
  join_mcc_bi=True,rejoin_col_strings_bi=True):
  
  new_merchant_col(df, old_col, col)
    
  if lowercase:
    lowercase_col(df, col)
  if split_string != '[]':
      split(df, col, split_string)
  if remove_string != '[]':
    remove_regex(df, col, remove_string)    
  if remove_empty_strings_bi:
    remove_empty_strings(df, col)
  if join_mcc_bi:
    join_mcc(df, col, 'mcc')
  if rejoin_col_strings_bi:
    rejoin_col_strings(df, col)
        
def clean_help():
  print("""The inputs are (df=data, old_col="merchant_string", col="merchant_string_clean",
    split_string=split_string, remove_string = remove_string,
    lowercase=True, remove_empty_strings_bi=True, join_mcc_bi=True,
    rejoin_col_strings_bi=True')

    REQUIRES the mcc column in df is called "mcc"

    df: is the data frame
    old_col: is the name of the column to clean - must be in the df
    col: is the name of the new col - need not be in the df'
    split_string: takes a regex to split string on to remove the split symbol and create divides in the string
    remove_string: takes a regex and removes everything that matches the regex
    join_mcc: joins the mcc column to the col selcted if mcc is not nan - mcc must be labeled "mcc"
    rejoin_col_strings_bi: set to true if you want a the cleaned column to be a string False will be a list""")
