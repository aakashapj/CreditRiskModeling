import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

df_model = pd.read_csv(
    'A:\\Practice_Project\\MachineLearningPractice\\CreditRiskModeling\\data\\processed\\df_encoded_V1_0.csv')

''' X and Y Dataset Creation '''
x_Dataset = df_model.drop(['Approved_Flag'], axis=1)
y_Dataset = df_model['Approved_Flag']
'''---------------------------------------------'''

labelEncoder = LabelEncoder()
y_DatasetEncoded = labelEncoder.fit_transform(y_Dataset)

''' Train and Test Split '''
x_train, x_test, y_train, y_test = train_test_split(x_Dataset, y_DatasetEncoded, test_size=0.2, random_state=42)
'''---------------------------------------------'''

''' Define the hyperparameters Grid '''
param_grid = {
    'colsample_bytree': [0.1, 0.3, 0.5],
    'learning_rate': [0.001, 0.1, 0.15],
    'max_depth': [2, 3, 4],
    'alpha': [1, 10, 100],
    'n_estimators': [10, 50, 100]
}
'''---------------------------------------------'''

''' Define the Answer Grid '''
answer_grid = {
    'combination': [],
    'train_Accuracy': [],
    'test_Accuracy': [],
    'colsample_bytree': [],
    'learning_rate': [],
    'max_depth': [],
    'alpha': [],
    'n_estimators': []
}
'''---------------------------------------------'''

''' XGBoost Model Training '''
index = 0
for colsample_bytree in param_grid['colsample_bytree']:
    for n_estimators in param_grid['n_estimators']:
        for learning_rate in param_grid['learning_rate']:
            for max_depht in param_grid['max_depth']:
                for alpha in param_grid['alpha']:
                    index += 1
                    xgbClassifier = xgb.XGBClassifier(objective='multi:softmax',
                                                      num_class=4,
                                                      learning_rate=learning_rate,
                                                      colsample_bytree=colsample_bytree,
                                                      max_depth=max_depht,
                                                      alpha=alpha,
                                                      n_estimators=n_estimators
                                                      )

                    # Model Training on Training Data
                    xgbClassifier.fit(x_train, y_train)

                    # Model Prediction
                    y_trainPred = xgbClassifier.predict(x_train)
                    y_testPred = xgbClassifier.predict(x_test)

                    # Accuracy of train and test result
                    train_Accuracy = accuracy_score(y_trainPred, y_train)
                    test_Accuracy = accuracy_score(y_testPred, y_test)

                    # Adding predicted value in the dictionary
                    answer_grid['combination'].append(index)
                    answer_grid['train_Accuracy'].append(round(train_Accuracy, 2))
                    answer_grid['test_Accuracy'].append(round(test_Accuracy, 2))
                    answer_grid['colsample_bytree'].append(colsample_bytree)
                    answer_grid['learning_rate'].append(learning_rate)
                    answer_grid['max_depth'].append(max_depht)
                    answer_grid['alpha'].append(alpha)
                    answer_grid['n_estimators'].append(n_estimators)

                    # Printing the result and values
                    print('combination: ', index)
                    print(f'colsample_bytree: {colsample_bytree}', f'learning_rate: {learning_rate}',
                          f'max_depth: {max_depht}', f'alpha: {alpha}', f'n_estimators: {n_estimators}')
                    print(f'train_Accuracy: {train_Accuracy:.2f}')
                    print(f'test_Accuracy: {test_Accuracy:.2f}')
                    print('-' * 30)
'''---------------------------------------------'''

'''
Best Combination

colsample_bytree: 0.5 
learning_rate: 0.15 
max_depth: 4 
alpha: 1 
n_estimators: 100
train_Accuracy: 0.80
test_Accuracy: 0.78

'''
'''---------------------------------------------'''


df_Answer = pd.DataFrame(answer_grid)

df_Answer.to_csv('A:\\Practice_Project\\MachineLearningPractice\\CreditRiskModeling\\reports\\answerGrid3.csv', index=False)
