import pandas as pd
from scipy.stats import chi2_contingency, f_oneway
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('A:\\Practice_Project\\MachineLearningPractice\\CreditRiskModeling\\data\\raw\\df_merged.csv')


# Analysis on Categorical Data Columns
significanceLevel = 0.05

ObjectColumn = []
for i in df.columns:
    if df[i].dtypes == object:
        ObjectColumn.append(i)

# ObjectColumn[:-1]
accepted_columns = []
for i in ObjectColumn[:-1]:
    chi2, pvalue, dof, _ = chi2_contingency(pd.crosstab(df[i], df[ObjectColumn[-1:][0]]))
    print("Chi-squared test:", "->", i, "->", pvalue)
    if pvalue < significanceLevel:
        accepted_columns.append(i)


# Analysis on Numerical Data Columns
NumericColumns = []
VIF_Threshold = 6
for i in df.columns[1:]:
    if df[i].dtypes != object:
        NumericColumns.append(i)

Num_accepted_columns = []
vif_data = df[NumericColumns]
total_columns = vif_data.shape[1]
column_index = 0

for i in range(0, total_columns):
    VIF_Value = variance_inflation_factor(vif_data, column_index)
    print(NumericColumns[i], "->", VIF_Value)

    if VIF_Value <= VIF_Threshold:
        Num_accepted_columns.append(NumericColumns[i])
        print(NumericColumns[i], "->", 'Accepted')
        column_index += 1
    else:
        vif_data = vif_data.drop([NumericColumns[i]], axis=1)
        print(NumericColumns[i], "->", 'Removed')


Num_columns_Final = []

for i in Num_accepted_columns:
    a = list(df[i])
    b = list(df['Approved_Flag'])

    group_P1 = [value for value, group in zip(a, b) if group == 'P1']
    group_P2 = [value for value, group in zip(a, b) if group == 'P2']
    group_P3 = [value for value, group in zip(a, b) if group == 'P3']
    group_P4 = [value for value, group in zip(a, b) if group == 'P4']

    f_stat, pvalue = f_oneway(group_P1, group_P2, group_P3, group_P4)

    if pvalue <= significanceLevel:
        Num_columns_Final.append(i)


# Categorical Columns Encoding
# Label Encoding of Ordinal data -> Education Column
labelEncoder = LabelEncoder()
df['EDUCATION'] = labelEncoder.fit_transform(df['EDUCATION'])
df['EDUCATION'].value_counts()

# One-Hot Encoding of Nominal Data -> Marital Status, Gender, Last_prod_enq2, First_prod_enq2
ObjectColumnEncode = [value for value in ObjectColumn if (value != 'Approved_Flag' and value != 'EDUCATION')]
df_Encoded = pd.get_dummies(df, columns=ObjectColumnEncode)

# Transferring Processed DataFrame to Processed Data Folder
df_Encoded.to_csv("A:\\Practice_Project\\MachineLearningPractice\\CreditRiskModeling\\data\\processed\\df_encoded.csv", index=False)
df_Encoded.to_csv("A:\\Practice_Project\\MachineLearningPractice\\CreditRiskModeling\\data\\processed\\df_encoded2.csv", index=False)