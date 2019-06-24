import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from utils import MultiColumnLabelEncoder
from sklearn.model_selection import train_test_split

# One hot feature expansion
def one_hot(filename1, filename2, keyword):
  if isinstance(filename1, str):
    data = pd.read_csv(filename1, delimiter=',')
    data = pd.get_dummies(data)
    data = data.pivot_table(index= [keyword], aggfunc=np.mean)
    data.to_csv(filename2)
  else:
    data = filename1
    data = pd.get_dummies(data)
    data.to_csv(filename2, index=False)
  

# Labencoder feature processing
def encoder(filename1, filename2, keyword):
  if isinstance(filename1, str):
    data = pd.read_csv(filename1, delimiter=',')
    title = data.columns.values.tolist()
    col_name = []
    for i in range(data.shape[1]):
      if isinstance(data.iloc[1, i], str):
        col_name.append(title[i])
    data = MultiColumnLabelEncoder(columns = col_name).fit_transform(data) 
    # data = data.pivot_table(index= [keyword], aggfunc=np.mean)
    data.to_csv(filename2)
  else:
    data = filename1
    data = pd.get_dummies(data)
    title = data.columns.values.tolist()
    col_name = []
    for i in range(data.shape[1]):
      if isinstance(data.iloc[1, i], str):
        col_name.append(title[i])
    data = MultiColumnLabelEncoder(columns = col_name).fit_transform(data) 
    data.to_csv(filename2, index= False)
  

# merge the dataset into one training set
def MERGE(filename1, filename2, mergekey):
  if isinstance(filename1,str):
    left = pd.read_csv(filename1, delimiter=',')
  else:
    left = filename1
  if isinstance(filename2, str):
    right = pd.read_csv(filename2, delimiter=',')
  else:
    right = filename2
  result = pd.merge(left, right, how='left', on=[mergekey])
  return result

def main():
  one_hot('../Data/raw_data/bureau_balance.csv', '../Data/one-hot/BB_unique.csv', "SK_ID_BUREAU")
  one_hot('../Data/raw_data/bureau.csv', '../Data/one-hot/B_unique.csv', "SK_ID_CURR")
  one_hot('../Data/raw_data/previous_application.csv', '../Data/one-hot/PA_unique.csv', "SK_ID_CURR")
  one_hot('../Data/raw_data/installments_payments.csv', '../Data/one-hot/IP_unique.csv', "SK_ID_PREV")
  one_hot('../Data/raw_data/POS_CASH_balance.csv', '../Data/one-hot/PCB_unique.csv', "SK_ID_PREV")
  one_hot('../Data/raw_data/credit_card_balance.csv', '../Data/one-hot/CCB_unique.csv', "SK_ID_PREV")
  encoder('../Data/raw_data/bureau_balance.csv', '../Data/encoder/BB_enc.csv', "SK_ID_BUREAU")
  encoder('../Data/raw_data/previous_application.csv', '../Data/encoder/PA_enc.csv', "SK_ID_CURR")
  encoder('../Data/raw_data/bureau.csv', '../Data/encoder/B_enc.csv', "SK_ID_CURR")
  encoder('../Data/raw_data/installments_payments.csv', '../Data/encoder/IP_enc.csv', "SK_ID_PREV")
  encoder('../Data/raw_data/POS_CASH_balance.csv', '../Data/encoder/PCB_enc.csv', "SK_ID_PREV")
  encoder('../Data/raw_data/credit_card_balance.csv', '../Data/encoder/CCB_enc.csv', "SK_ID_PREV")
  dataset = MERGE('../Data/raw_data/application_train.csv', '../Data/one-hot/B_unique.csv','SK_ID_CURR')
  dataset = MERGE(dataset,'../Data/one-hot/BB_unique.csv','SK_ID_BUREAU')
  dataset = MERGE(dataset, '../Data/one-hot/PA_unique.csv', 'SK_ID_CURR')
  dataset = MERGE(dataset,'../Data/one-hot/PCB_unique.csv', 'SK_ID_PREV')
  dataset = MERGE(dataset, '../Data/one-hot/IP_unique.csv','SK_ID_PREV')
  dataset = MERGE(dataset, '../Data/one-hot/CCB_unique.csv','SK_ID_PREV')
  encoder('../Data/DATAENC.csv', '../Data/DATA.csv', "SK_ID_CURR")
  dataset = pd.read_csv('../Data/DATA.csv', delimiter=',')
  dataset.drop([0], axis= 0, inplace= True)
  label = dataset.TARGET
  dataset.drop(columns=['TARGET'],inplace= True)
  X_train, X_test, y_train, y_test = train_test_split(dataset, label, test_size=0.25)
  y_test.to_csv('../Data/label_enc.csv', index = False)
  X_test.to_csv('../Data/train_enc.csv', index = False)
  # dataset.to_csv('../Data/DATAENC.csv', index = False)
  return

if __name__ == '__main__':
  main()
