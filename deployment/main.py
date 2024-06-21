import streamlit as st
import pickle
import numpy as np
import os

sys_cwd = os.getcwd()

os.chdir(sys_cwd + "\\deployment")

# Page Configuration
st.set_page_config(layout='wide')


# Model load from pickle file
modelFile = open(os.getcwd() + "\\crmodel.sav", 'rb')
model = pickle.load(modelFile)
# -----------------------------------------------

# Model Prediction

def modelPredict(valueInputs):
    inputValue_np = np.array(valueInputs, dtype=object)

    predicted = model.predict(inputValue_np.reshape(1, -1))

    return predicted[0]


# Web App section
st.header('Credit Risk Prediction system', divider='rainbow', anchor=False)

col1, col2, col3, col4 = st.columns(4)
valueInput = [0]*55
with col1:
    valueInput[1] = st.number_input("A/c Open in 6M (%)", min_value=0.0)
    valueInput[2] = st.number_input("A/c Close in 6M (%)", min_value=0)
    valueInput[3] = st.number_input("A/c Total Close in 6M", value=0, min_value=0)
    valueInput[4] = st.number_input("A/c Close in 12M (%)", min_value=0.0)
    valueInput[5] = st.number_input("Total Missed Payments", value=0, min_value=0)
    valueInput[6] = st.number_input("Count of Credit card ", value=0, min_value=0)
    valueInput[7] = st.number_input("Count of Housing loan", value=0, min_value=0)
    valueInput[8] = st.number_input("Count of Personal loan", value=0, min_value=0)
    valueInput[9] = st.number_input("Count of secured loan", value=0, min_value=0)
    valueInput[10] = st.number_input("Count of unsecured loan", value=0, min_value=0)
    valueInput[11] = st.number_input("Count of other loan", value=0, min_value=0)

with col2:
    valueInput[12] = st.number_input("Count of oldest loan", value=0, min_value=0)
    valueInput[13] = st.number_input("Count of newest loan", value=0, min_value=0)
    valueInput[14] = st.number_input("Time Since recent Payment made", value=0, min_value=0)
    valueInput[15] = st.number_input("Maximum recent level of delinquency", value=0, min_value=0)
    valueInput[16] = st.number_input("Number of times delinquent b/w 6M-12M", value=0, min_value=0)
    valueInput[17] = st.number_input("Number of times 60+ dpd", value=0, min_value=0)
    valueInput[18] = st.number_input("Number of standard Payments in 12M", value=0, min_value=0)
    valueInput[19] = st.number_input("Number of standard Payments", value=0, min_value=0)
    valueInput[20] = st.number_input("Number of sub standard payments in 6M", value=0, min_value=0)
    valueInput[21] = st.number_input("Number of sub standard payments in 12M", value=0, min_value=0)
    valueInput[22] = st.number_input("Number of doubtful payments", value=0, min_value=0)

with col3:
    valueInput[23] = st.number_input("Number of doubtful payments 12M", value=0, min_value=0)
    valueInput[24] = st.number_input("Number of loss accounts", value=0, min_value=0)
    valueInput[25] = st.number_input("Recent level of delinquency", value=0, min_value=0)
    valueInput[26] = st.number_input("Credit card enquiries in 12M", value=0, min_value=0)
    valueInput[27] = st.number_input("Personal Loan enquiries in 12M", value=0, min_value=0)
    valueInput[28] = st.number_input("Time since recent enquiry", value=0, min_value=0)
    valueInput[29] = st.number_input("Enquiries in 3M", value=0, min_value=0)
    valueInput[30] = st.number_input("NET MONTHLY INCOME", value=0, min_value=0)
    valueInput[31] = st.number_input("Time with current Employer", value=0, min_value=0)
    valueInput[32] = st.number_input("Credit card Flag", min_value=0)
    valueInput[33] = st.number_input("Personal Loan Flag", min_value=0)

with col4:
    valueInput[34] = st.number_input("Percent enquiries PL in 6M", min_value=0.0)
    valueInput[35] = st.number_input("Percent enquiries CC in 6M", min_value=0.0)
    valueInput[36] = st.number_input("Housing Loan Flag", min_value=0)
    valueInput[37] = st.number_input("Gold Loan Flag", min_value=0)

    # Education Input Formatting
    DicEducation = {"SSC":1,
                    "12TH":2,
                    "UNDER GRADUATE":3,
                    "GRADUATE":4,
                    "POST-GRADUATE":5,
                    "PROFESSIONAL":6,
                    "OTHERS":0}
    Education = st.selectbox(label='Education',
                             options=["SSC", "12TH", "UNDER GRADUATE", "GRADUATE", "POST-GRADUATE", "PROFESSIONAL",
                                      "OTHERS"])

    valueInput[38] = DicEducation[Education]

    # Marital Status Input Formatting
    maritalStatus = st.selectbox(label='Marital Status',
                                 options=["MARRIED", "SINGLE"])

    valueInput[39] = False
    valueInput[40] = False

    if maritalStatus == "MARRIED":
        valueInput[39] = True
    else:
        valueInput[40] = True

    # Gender Status Input Formatting
    valueInput[41] = False
    valueInput[42] = False

    Gender = st.selectbox(label='Gender',
                          options=["MALE", "FEMALE"])

    if Gender == "FEMALE":
        valueInput[41] = True
    else:
        valueInput[42] = True

    # Last Product Enquired Input Formatting
    valueInput[43] = False
    valueInput[44] = False
    valueInput[45] = False
    valueInput[46] = False
    valueInput[47] = False
    valueInput[48] = False

    lastProductEnq = st.selectbox(label='Last Product Enquired',
                                  options=["AL", "CC", "Consumer", "HL", "PL", "OTHER"])

    if lastProductEnq == "AL":
        valueInput[43] = True
    elif lastProductEnq == "CC":
        valueInput[44] = True
    elif lastProductEnq == "Consumer":
        valueInput[45] = True
    elif lastProductEnq == "HL":
        valueInput[46] = True
    elif lastProductEnq == "PL":
        valueInput[47] = True
    elif lastProductEnq == "OTHER":
        valueInput[48] = True

    # First Product Enquired Input Formatting
    valueInput[49] = False
    valueInput[50] = False
    valueInput[51] = False
    valueInput[52] = False
    valueInput[53] = False
    valueInput[54] = False

    firstProductEnq = st.selectbox(label='First Product Enquired',
                                   options=["AL", "CC", "Consumer", "HL", "PL", "OTHER"])

    if lastProductEnq == "AL":
        valueInput[49] = True
    elif lastProductEnq == "CC":
        valueInput[50] = True
    elif lastProductEnq == "Consumer":
        valueInput[51] = True
    elif lastProductEnq == "HL":
        valueInput[52] = True
    elif lastProductEnq == "PL":
        valueInput[53] = True
    elif lastProductEnq == "OTHER":
        valueInput[54] = True


st.header('', divider='rainbow', anchor=False)
predicted = 4
if st.button("Check Risk"):
    predicted = modelPredict(valueInput[1:])

risk_Index = ''
match predicted:
    case 0:
        risk_Index = 'P1 Low Risk Customer'
    case 1:
        risk_Index = 'P2 Low-medium Risk Customer'
    case 2:
        risk_Index = 'P3 Medium-High Risk Customer'
    case 3:
        risk_Index = 'P4 High Risk Customer'


st.header(f'{risk_Index} ', anchor=False)
st.header('', divider='rainbow', anchor=False)
# -----------------------------------------------
