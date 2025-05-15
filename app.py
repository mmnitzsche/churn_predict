# app.py

import streamlit as st
import pandas as pd
import pickle
import numpy as np
from utils.input_transformer import stepEnconding

# Carrega o DataFrame base
df = pd.read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Carrega o modelo treinado
with open("model_churn.pkl", "wb") as f:
    modelo = pickle.load(f)

# Define colunas categóricas e numéricas
obj_cols = df.select_dtypes(include="object").columns.tolist()
num_cols = df.select_dtypes(include=np.number).columns.tolist()

for col in ["customerID"]:
    if col in obj_cols:
        obj_cols.remove(col)
    if col in num_cols:
        num_cols.remove(col)

st.title("Churn Prediction")

with st.form("form_churn"):
    st.subheader("Fill the client data")
    
    col1, col2 = st.columns(2)

    input_obj = {}
    input_num = {}

    for i, col in enumerate(obj_cols):
        options = df[col].dropna().unique().tolist()
        with (col1 if i % 2 == 0 else col2):
            selected = st.selectbox(f"{col}:", options, key=col)
            input_obj[col] = selected

    for i, col in enumerate(num_cols):
        with (col1 if i % 2 == 0 else col2):
            val = st.number_input(f"{col}:", min_value=0.0, value=float(df[col].mean()), key=col)
            input_num[col] = val

    submitted = st.form_submit_button("Prever churn")

if submitted:
    input_data = {**input_obj, **input_num}
    input_df = pd.DataFrame([input_data])

    # Alinha com colunas do treino
    df_dummies = stepEnconding(df, input_df)

    # Mostrar dados processados
    st.subheader("Dados processados para o modelo (df_dummies)")
    st.dataframe(input_df)

    # Predição
    pred = modelo.predict(df_dummies)[0]
    prob = modelo.predict_proba(df_dummies)[0][1]

    st.success(f"Resultado: {'Vai churnar' if pred == 1 else 'Não vai churnar'}")
    st.write(f"Probabilidade de churn: {prob:.2%}")
