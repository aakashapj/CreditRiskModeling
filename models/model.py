import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

df_model = pd.read_csv(
    'A:\\Practice_Project\\MachineLearningPractice\\CreditRiskModeling\\data\\processed\\df_encoded_V1_0.csv')

''' X and Y Dataset Creation '''
X_Dataset = df_model.drop(['Approved_Flag'], axis=1)
Y_Dataset = df_model['Approved_Flag']
'''---------------------------------------------'''

''' Train and Test Dataset Split '''
# Train and Test split of Dataset
x_train, x_test, y_train, y_test = train_test_split(X_Dataset, Y_Dataset, test_size=0.2, random_state=42)
'''---------------------------------------------'''

X_Dataset.columns.duplicated()


'''  Random Forest Classifier '''
# Model Fitting Using Random Forest Classifier
rf_Classifier = RandomForestClassifier(n_estimators=200, random_state=42)
rf_Classifier.fit(x_train, y_train)

# Model Prediction
y_pred = rf_Classifier.predict(x_test)

# checking prediction Metrics
accuracy = accuracy_score(y_test, y_pred)
print(f'Model Accuracy: {accuracy}')
print()
precision, recall, fscore, _ = precision_recall_fscore_support(y_pred, y_test)

for i, v in enumerate(['P1', 'P2', 'P3', 'P4']):
    print(f'Class {v}:')
    print(f'Precision: {precision[i]}')
    print(f'Recall: {recall[i]}')
    print(f'F-score: {fscore[i]}')
    print()

# Random Forest Accuracy -> 0.76500
'''------------------------------------------------'''

''' XGBoost Classifier '''

# y_test encoding as xgBoost accepts only numerical data
labelEncoder = LabelEncoder()
y_trainEncoded = labelEncoder.fit_transform(y_train)
y_testEncoded = labelEncoder.fit_transform(y_test)

# Model Fitting Using XGBoost Classifier
xgb_Classifier = xgb.XGBClassifier(objective='multi:softmax', num_class=4)
xgb_Classifier.fit(x_train, y_trainEncoded)

# Model Prediction
y_pred = xgb_Classifier.predict(x_test)

# checking prediction Metrics
accuracy = accuracy_score(y_pred, y_testEncoded)
print(f'Model Accuracy: {accuracy}')
print()

precision, recall, fscore, _ = precision_recall_fscore_support(y_testEncoded, y_pred)
for i, v in enumerate(['P1', 'P2', 'P3', 'P4']):
    print(f'Class {v}:')
    print(f'Precision: {precision[i]}')
    print(f'Recall: {recall[i]}')
    print(f'F-score: {fscore[i]}')
    print()

# XGBoost Accuracy -> 0.7733
'''------------------------------------------------'''

''' Decision Tree Classifier '''

# Model Fitting Using XGBoost Classifier
DecisionTreeModel = DecisionTreeClassifier(max_depth=20, min_samples_split=10)
DecisionTreeModel.fit(x_train, y_train)

# Model Prediction
y_pred = DecisionTreeModel.predict(x_test)

# Checking Model Prediction
accuracy = accuracy_score(y_test, y_pred)
print(f'Model Accuracy: {accuracy}')
print()

precision, recall, fscore, _ = precision_recall_fscore_support(y_test, y_pred)

for i, v in enumerate(['P1', 'P2', 'P3', 'P4']):
    print(f'Class {v}:')
    print(f'Precision: {precision[i]}')
    print(f'Recall: {recall[i]}')
    print(f'F-Score: {fscore[i]}')
    print()

# Decision Tree Classifier Accuracy -> 0.7100
'''------------------------------------------------'''