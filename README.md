# 📊 Customer Churn Prediction (Streamlit Deployed)

A machine learning–based web application that predicts whether a customer is likely to churn based on historical customer data.  
This project demonstrates the complete machine learning lifecycle — from data preprocessing and model training to deployment using Streamlit.

---

## 🚀 Live Application

The project is deployed using Streamlit and allows users to input customer details to receive real-time churn predictions through an interactive web interface.
https://ann-classification-churn-w7wakzbi7f7g78lrb9wkgn.streamlit.app/
---

## 📌 Problem Statement

Customer churn is a major concern for banking and subscription-based businesses, as acquiring new customers is often more expensive than retaining existing ones.  
The goal of this project is to predict customer churn using historical data so that businesses can take proactive retention actions.

---

## 🧠 Model Overview

- Deep Learning–based classification model built using TensorFlow / Keras
- Trained on structured customer data
- Outputs a churn probability score between 0 and 1

### 🔍 Churn Risk Interpretation
- Low Risk → Probability < 0.40  
- Medium Risk → 0.40 ≤ Probability ≤ 0.70  
- High Risk → Probability > 0.70  

---

## 📊 Dataset Information

- File: Churn_Modelling.csv
- Contains demographic and behavioral features such as:
  - Credit Score
  - Geography
  - Gender
  - Age
  - Tenure
  - Account Balance
  - Number of Products
  - Credit Card Status
  - Active Membership
  - Estimated Salary

The dataset is used for educational and learning purposes.

---

## 🏗️ Project Structure

churn-prediction/
├── app.py
├── Churn_Modelling.csv
├── model_training.ipynb
├── hyperparametertuning.ipynb
├── prediction.ipynb
├── requirements.txt
├── model.h5
├── le.pkl
├── oe.pkl
├── sc.pkl
└── README.md

---

## ⚙️ Tech Stack

- Python
- TensorFlow / Keras
- Scikit-learn
- Pandas
- NumPy
- Matplotlib
- Streamlit

---

## 🔄 Machine Learning Workflow

1. Data loading and inspection  
2. Data preprocessing and cleaning  
3. Feature encoding (Label Encoding & One-Hot Encoding)  
4. Feature scaling using StandardScaler  
5. Model training using a neural network  
6. Hyperparameter tuning  
7. Model evaluation  
8. Real-time prediction using Streamlit  

---

## ▶️ How to Run Locally

1. Clone the repository

git clone https://github.com/aniketsingla01/ANN-Classification-Churn.git  

2. Create and activate a virtual environment

python -m venv venv  
source venv/bin/activate  
(Windows: venv\Scripts\activate)

3. Install dependencies

pip install -r requirements.txt

4. Run the Streamlit application

streamlit run app.py

---

## 📈 Output

The application displays:
- Churn probability score
- Risk category (Low / Medium / High)

This enables easy interpretation for non-technical users.

---

Developed and deployed a customer churn prediction system using deep learning and Streamlit, covering end-to-end ML workflow from data preprocessing and model training to real-time web-based predictions.
