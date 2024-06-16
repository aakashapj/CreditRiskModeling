"""
Best Combination

colsample_bytree: 0.5 
learning_rate: 0.15 
max_depth: 4 
alpha: 1 
n_estimators: 100
train_Accuracy: 0.80
test_Accuracy: 0.78
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder

df_model = pd.read_csv(
    'A:\\Practice_Project\\MachineLearningPractice\\CreditRiskModeling\\data\\processed\\df_encoded_V1_0.csv')

''' X and Y Dataset Creation '''
x_Dataset = df_model.drop(['Approved_Flag'], axis=1)
y_Dataset = df_model['Approved_Flag']
'''---------------------------------------------'''

''' Y Dataset Label Encoding for XGBoost '''
labelEncoder = LabelEncoder()
y_DatasetEncoded = labelEncoder.fit_transform(y_Dataset)
'''---------------------------------------------'''

''' Creating Dataset from unseen Dataset '''
df_unseen = pd.read_csv('A:\\Practice_Project\\MachineLearningPractice\\CreditRiskModeling\\data\\raw\\Unseen.csv')
'''---------------------------------------------'''

nonNumCol = []
for col in df_unseen.columns:
    if df_unseen[col].dtypes == 'object':
        nonNumCol.append(col)

''' Encoding of Non-Numeric Columns '''

''' Label Encoding '''
df_unseen.loc[df_unseen['EDUCATION'] == 'OTHERS', ['EDUCATION']] = 1
df_unseen.loc[df_unseen['EDUCATION'] == 'SSC', ['EDUCATION']] = 2
df_unseen.loc[df_unseen['EDUCATION'] == '12TH', ['EDUCATION']] = 3
df_unseen.loc[df_unseen['EDUCATION'] == 'UNDER GRADUATE', ['EDUCATION']] = 4
df_unseen.loc[df_unseen['EDUCATION'] == 'GRADUATE', ['EDUCATION']] = 5
df_unseen.loc[df_unseen['EDUCATION'] == 'POST-GRADUATE', ['EDUCATION']] = 6

''' Column datatype conversion from object to int'''
df_unseen['EDUCATION'] = df_unseen['EDUCATION'].astype(int)

''' One-hot Encoding'''
df_unseenEncode = pd.get_dummies(df_unseen, columns=nonNumCol.remove('EDUCATION'))

df_unseenEncode.to_csv("A:\\Practice_Project\\MachineLearningPractice\\CreditRiskModeling\\data\\processed"
                       "\\df_unseenEncoded.csv", index=False)

'''---------------------------------------------------------'''

''' Prediction Model Creation '''

xgbClassifier = xgb.XGBClassifier(objective='multi:softmax',
                                  num_class=4,
                                  colsample_bytree=0.5,
                                  learning_rate=0.15,
                                  max_depth=4,
                                  n_estimators=100,
                                  alpha=1)

''' Model train'''
xgbClassifier.fit(x_Dataset, y_DatasetEncoded)

''' Prediction on Unseen Data'''
y_unseenPred = xgbClassifier.predict(df_unseenEncode)

''' Merging the unseen training data and predicted output'''
df_pred = df_unseenEncode.copy()
df_pred['Target'] = y_unseenPred

'''Preparing model using pickle library for deployment'''

import pickle

filename = "A:\\Practice_Project\\MachineLearningPractice\\CreditRiskModeling\\deployment\\crmodel.sav"
fileHandler = open(filename, 'wb')
pickle.dump(xgbClassifier, fileHandler)
fileHandler.close()
