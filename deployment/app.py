import streamlit as st
import pickle
import numpy as np

st.set_page_config(layout='wide')

modelFile = open("A:\\Practice_Project\\MachineLearningPractice\\CreditRiskModeling\\deployment\\crmodel.sav", 'rb')
model = pickle.load(modelFile)

SingleInput = np.array(
    [0.0, 0.0, 0, 0.0, 0, 0, 0, 4, 1, 4, 0, 72, 18, 549, 29, 0, 0, 11, 0, 0, 0, 0, 0, 0, 29, 0, 0, 566, 0, 51000, 114,
     0, 1, 0.0, 0.0, 1, 0, 3, True, False, False, True, False, False, False, False, True, False, False, False, False,
     False, True, False])

model.predict(SingleInput.reshape(1, -1))
value = 3

st.header('Credit Risk Prediction system', divider='rainbow', anchor=False)
st.button('Predict Customer Priority')
st.subheader(f'Answer: {value}', anchor=False)

col1, col2, col3, col4 = st.columns(4)

with col1:
    input1 = st.number_input("A/c Open in 6M (%)", min_value=0.0)
    input2 = st.number_input("A/c Close in 6M (%)", min_value=0)
    input3 = st.number_input("A/c Total Close in 6M", value=0, min_value=0)
    input4 = st.number_input("A/c Close in 12M (%)", min_value=0)
    input5 = st.number_input("Total Missed Payments", value=0, min_value=0)
    input6 = st.number_input("Count of Credit card ", value=0, min_value=0)
    input7 = st.number_input("Count of Housing loan", value=0, min_value=0)
    input8 = st.number_input("Count of Personal loan", value=0, min_value=0)
    input9 = st.number_input("Count of secured loan", value=0, min_value=0)
    input10 = st.number_input("Count of unsecured loan", value=0, min_value=0)

with col2:
    input11 = st.number_input("Count of other loan", value=0, min_value=0)
    input12 = st.number_input("Count of oldest loan", value=0, min_value=0)
    input13 = st.number_input("Count of newest loan", value=0, min_value=0)
    input14 = st.number_input("Time Since recent Payment made", value=0, min_value=0)
    input15 = st.number_input("Maximum recent level of delinquency", value=0, min_value=0)
    input16 = st.number_input("Number of times delinquent b/w 6M-12M", value=0, min_value=0)
    input17 = st.number_input("Number of times 60+ dpd", value=0, min_value=0)
    input18 = st.number_input("Number of standard Payments in 12M", value=0, min_value=0)
    input19 = st.number_input("Number of standard Payments", value=0, min_value=0)
    input20 = st.number_input("Number of sub standard payments in 6M", value=0, min_value=0)

with col3:
    input21 = st.number_input("Number of sub standard payments in 12M", value=0, min_value=0)
    input22 = st.number_input("Number of doubtful payments", value=0, min_value=0)
    input23 = st.number_input("Number of doubtful payments 12M", value=0, min_value=0)
    input24 = st.number_input("Number of loss accounts", value=0, min_value=0)
    input25 = st.number_input("Recent level of delinquency", value=0, min_value=0)
    input26 = st.number_input("Credit card enquiries in 12M", value=0, min_value=0)
    input27 = st.number_input("Personal Loan enquiries in 12M", value=0, min_value=0)
    input28 = st.number_input("Time since recent enquiry", value=0, min_value=0)
    input29 = st.number_input("Enquiries in 3M", value=0, min_value=0)
    input30 = st.number_input("NET MONTHLY INCOME", value=0, min_value=0)

with col4:
    input31 = st.number_input("Time with current Employer", value=0, min_value=0)
    input32 = st.number_input("Credit card Flag", min_value=0)
    input33 = st.number_input("Personal Loan Flag", min_value=0)
    input34 = st.number_input("Percent enquiries PL in 6M", min_value=0)
    input35 = st.number_input("Percent enquiries CC in 6M", min_value=0)
    input36 = st.number_input("Housing Loan Flag", min_value=0)
    input37 = st.number_input("Gold Loan Flag", min_value=0)
