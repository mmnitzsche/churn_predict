import streamlit as st
import pandas as pd
import pickle  # para carregar modelo
import numpy as np


# Carrega o DataFrame
df = pd.read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Carregando o modelo treinado
with open("modelo_churn.pkl", "rb") as f:
    modelo = pickle.load(f)

# Separar colunas por tipo
obj_cols = df.select_dtypes(include="object").columns.tolist()
num_cols = df.select_dtypes(include=np.number).columns.tolist()

# Remover colunas desnecessárias
for col in ["customerID", "Churn"]:
    if col in obj_cols:
        obj_cols.remove(col)
    if col in num_cols:
        num_cols.remove(col)

st.title("Churn Prediction")


# Formulário para novo cliente
with st.form("form_churn"):
    st.subheader("Fill the client data")
    
    col1, col2 = st.columns(2)

    input_obj = {}
    input_num = {}

    # Campos categóricos em colunas alternadas
    for i, col in enumerate(obj_cols):
        options = df[col].dropna().unique().tolist()
        with (col1 if i % 2 == 0 else col2):
            selected = st.selectbox(f"{col}:", options, key=col)
            input_obj[col] = selected

    # Campos numéricos em colunas alternadas
    for i, col in enumerate(num_cols):
        with (col1 if i % 2 == 0 else col2):
            val = st.number_input(f"{col}:", min_value=0.0, value=float(df[col].mean()), key=col)
            input_num[col] = val

    submitted = st.form_submit_button("Prever churn")

if submitted:
    # Juntar os dados
    input_data = {**input_obj, **input_num}
    input_df = pd.DataFrame([input_data])

    # Criar dummies
    df_dummies = pd.get_dummies(input_df, drop_first=True)
    st.dataframe(df_dummies)

    # Garantir as colunas do modelo
    modelo_cols = modelo.feature_names_in_
    for col in modelo_cols:
        if col not in df_dummies.columns:
            df_dummies[col] = 0

    # Reordenar as colunas
    df_dummies = df_dummies[modelo_cols]

    # Fazer predição
    pred = modelo.predict(df_dummies)[0]
    prob = modelo.predict_proba(df_dummies)[0][1]

    st.success(f"Resultado: {'Vai churnar' if pred == 1 else 'Não vai churnar'}")
    st.write(f"Probabilidade de churn: {prob:.2%}")
