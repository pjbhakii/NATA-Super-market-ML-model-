import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json

st.set_page_config(page_title='NATA Campaign Targeting', layout='centered')
st.title('NATA Supermarket - Campaign Targeting')

model = joblib.load('ada_model.pkl')
scaler = joblib.load('scaler.pkl')
kmeans = joblib.load('kmeans.pkl')
info = json.load(open('model_info.json'))

st.header('Customer Profile')
col1, col2 = st.columns(2)
with col1:
    Income = st.number_input('Annual Income', min_value=0, max_value=150000, value=52000, step=1000)
    Age = st.slider('Age', 18, 90, 45)
    Education = st.selectbox('Education', ['Basic', '2n Cycle', 'Graduation', 'Master', 'PhD'], index=2)
    Marital = st.selectbox('Marital Status', ['Married', 'Together', 'Single', 'Divorced', 'Widow'])
with col2:
    Kidhome = st.number_input('Kids at home', 0, 3, 0)
    Teenhome = st.number_input('Teens at home', 0, 3, 0)
    Recency = st.slider('Days since last purchase', 0, 99, 30)
    Tenure = st.slider('Days as customer', 0, 700, 350)

st.header('Spending (last 2 years)')
col3, col4 = st.columns(2)
with col3:
    MntWines = st.number_input('Wines', 0, 2000, 300)
    MntFruits = st.number_input('Fruits', 0, 500, 25)
    MntMeatProducts = st.number_input('Meat', 0, 2000, 160)
with col4:
    MntFishProducts = st.number_input('Fish', 0, 500, 35)
    MntSweetProducts = st.number_input('Sweets', 0, 500, 25)
    MntGoldProds = st.number_input('Gold', 0, 500, 45)

st.header('Purchase Behaviour')
col5, col6 = st.columns(2)
with col5:
    NumWebPurchases = st.number_input('Web purchases', 0, 30, 4)
    NumCatalogPurchases = st.number_input('Catalog purchases', 0, 30, 2)
    NumStorePurchases = st.number_input('Store purchases', 0, 30, 5)
with col6:
    NumDealsPurchases = st.number_input('Deal purchases', 0, 20, 2)
    NumWebVisitsMonth = st.number_input('Web visits per month', 0, 20, 5)
    Complain = st.checkbox('Complained in last 2 years')

st.header('Past Campaigns Accepted')
accepted = [st.checkbox(f'Campaign {i}') for i in range(1, 6)]

edu_map = {'Basic': 1, '2n Cycle': 2, 'Graduation': 3, 'Master': 4, 'PhD': 5}

row = dict(info['defaults'])
row.update({
    'Education': edu_map[Education], 'Income': Income, 'Age': Age,
    'Kidhome': Kidhome, 'Teenhome': Teenhome, 'Recency': Recency, 'Tenure': Tenure,
    'MntWines': MntWines, 'MntFruits': MntFruits, 'MntMeatProducts': MntMeatProducts,
    'MntFishProducts': MntFishProducts, 'MntSweetProducts': MntSweetProducts, 'MntGoldProds': MntGoldProds,
    'NumWebPurchases': NumWebPurchases, 'NumCatalogPurchases': NumCatalogPurchases,
    'NumStorePurchases': NumStorePurchases, 'NumDealsPurchases': NumDealsPurchases,
    'NumWebVisitsMonth': NumWebVisitsMonth, 'Complain': int(Complain)
})
for i in range(1, 6):
    row[f'AcceptedCmp{i}'] = int(accepted[i - 1])
row['Children'] = Kidhome + Teenhome
row['TotalSpend'] = MntWines + MntFruits + MntMeatProducts + MntFishProducts + MntSweetProducts + MntGoldProds
row['TotalPurchases'] = NumWebPurchases + NumCatalogPurchases + NumStorePurchases
row['TotalAccepted'] = sum(accepted)
for m in ['Married', 'Single', 'Together', 'Widow']:
    row[f'Marital_Status_{m}'] = int(Marital == m)

if st.button('Predict Response'):
    X_new = pd.DataFrame([row])[info['features']]
    prob = model.predict_proba(X_new)[0][1]

    seg_input = pd.DataFrame([row])
    seg_input['Spend_log'] = np.log1p(row['TotalSpend'])
    cluster = kmeans.predict(scaler.transform(seg_input[info['seg_cols']]))[0]
    segment = info['segments'][str(cluster)]

    st.subheader('Result')
    c1, c2 = st.columns(2)
    c1.metric('Response score', f'{prob:.2f}')
    c2.metric('Customer segment', segment)

    if prob >= info['threshold']:
        st.success(f'CONTACT this customer (score above {info["threshold"]:.2f} cut-off)')
    else:
        st.error(f'Do NOT contact (score below {info["threshold"]:.2f} cut-off)')
