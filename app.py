import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle

model = tf.keras.models.load_model('model.h5')

with open('le.pkl', 'rb') as file:
    le = pickle.load(file)

with open('oe.pkl', 'rb') as file:
    oe = pickle.load(file)

with open('sc.pkl', 'rb') as file:
    sc = pickle.load(file)


## streamlit app
st.title('Customer Churn PRediction')

geography = st.selectbox('Geography', oe.categories_[0])
gender = st.selectbox('Gender', le.classes_)
age = st.slider('Age', 18, 92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('Number of Products', 1, 4)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])

input_data = pd.DataFrame({
        'CreditScore': [credit_score],
    'Gender': [le.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

geo = oe.transform([[geography]])
geo_df = pd.DataFrame(geo, columns=oe.get_feature_names_out(['Geography']))

input_data = pd.concat([input_data.reset_index(drop= True), geo_df], axis = 1)
input_data_sc = sc.transform(input_data)

prediction = model.predict(input_data_sc)
pred_prob = prediction[0][0]

st.write(f'Churn Probability: {pred_prob:.2f}')

if pred_prob > 0.7:
    st.error("High churn risk")
elif pred_prob > 0.4:
    st.warning("Medium churn risk")
else:
    st.success("Low churn risk")
