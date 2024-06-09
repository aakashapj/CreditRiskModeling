import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

df_model = pd.read_csv(
    'A:\\Practice_Project\\MachineLearningPractice\\CreditRiskModeling\\data\\processed\\df_encoded_V1_0.csv')

''' X and Y Dataset Creation '''
x_Dataset = df_model.drop(['Approved_Flag'], axis=1)
y_Dataset = df_model['Approved_Flag']
'''---------------------------------------------'''

''' Train and Test Split '''
x_train, x_test, y_train, y_test = train_test_split(x_Dataset, y_Dataset, test_size=0.2, random_state=42)
'''---------------------------------------------'''




