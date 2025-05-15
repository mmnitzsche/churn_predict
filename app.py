import streamlit as st
import pandas as pd
import pickle
from utils.input_transformer import stepEnconding

df = pd.read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")
modelo = pickle.load(open("model_churn.pkl", "rb"))


def prever_churn(input_data_dict):
    input_df = pd.DataFrame([input_data_dict])
    df_dummies = stepEnconding(df, input_df)
    pred = modelo.predict(df_dummies)[0]
    prob = modelo.predict_proba(df_dummies)[0][1]
    return pred, prob, input_df


planos = {
    "Basic Plan": {
        "customerID": "5150-ITWWB",
        "gender": "Male",
        "SeniorCitizen": 1,
        "Partner": 0,
        "Dependents": 0,
        "tenure": 0,
        "PhoneService": 1,
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "Yes",
        "StreamingTV": "No",
        "StreamingMovies": "Yes",
        "Contract": "Month-to-month",
        "PaperlessBilling": 1,
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 94.85,
        "TotalCharges": 60,
        "Churn": "No",
        "years": 0,
    },
    "Silver Plan": {
        "customerID": "2002-EFGH",
        "gender": "Male",
        "SeniorCitizen": 1,
        "Partner": 0,
        "Dependents": 0,
        "tenure": 12,
        "PhoneService": 1,
        "MultipleLines": "Yes",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "Yes",
        "OnlineBackup": "Yes",
        "DeviceProtection": "Yes",
        "TechSupport": "Yes",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "One year",
        "PaperlessBilling": 1,
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 89.1,
        "TotalCharges": 1000,
        "Churn": "No",
        "years": 1,
    },
    "Gold Plan": {
        "customerID": "3003-IJKL",
        "gender": "Male",
        "SeniorCitizen": 1,
        "Partner": 0,
        "Dependents": 0,
        "tenure": 24,
        "PhoneService": 1,
        "MultipleLines": "Yes",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "Yes",
        "OnlineBackup": "Yes",
        "DeviceProtection": "Yes",
        "TechSupport": "Yes",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "Month-to-month",
        "PaperlessBilling": 1,
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 94.85,
        "TotalCharges": 60,
        "Churn": "No",
        "years": 2,
    },
}

cores_planos = {
    "Basic Plan": "#ffffff",  # verde
    "Silver Plan": "#bababa",  # azul
    "Gold Plan": "#e7d696",  # amarelo
}

description = {
    "Basic Plan": [
        "❌ Dependents included",
        "❌ Extended tenure",
        "✅ Phone service",
        "❌ Multiple lines",
        "✅ Internet service",
        "❌ Online security",
        "❌ Online backup",
        "❌ Device protection",
        "❌ Tech support",
        "❌ Streaming TV",
        "❌ Streaming movies",
    ],
    "Silver Plan": [
        "✅ Dependents included",
        "✅ Medium tenure",
        "✅ Phone service",
        "✅ Multiple lines",
        "✅ Internet service",
        "✅ Online security",
        "✅ Online backup",
        "❌ Device protection",
        "❌ Tech support",
        "✅ Streaming TV",
        "✅ Streaming movies",
    ],
    "Gold Plan": [
                "✅ Dependents included",
        "✅ Medium tenure",
        "✅ Phone service",
        "✅ Multiple lines",
        "✅ Internet service",
        "✅ Online security",
        "✅ Online backup",
        "✅ Device protection",
        "✅ Tech support",
        "✅ Streaming TV",
        "✅ Streaming movies",
    ]
}



st.title("Flexible Pricing Plans")
st.markdown("Escolha um plano para simular o risco de churn do cliente correspondente.")

col1, col2, col3 = st.columns(3)

for col, (plan_name, data) in zip([col1, col2, col3], planos.items()):
    cor = cores_planos[plan_name]
    description_desc = description[plan_name]

    pred, prob, input_df = prever_churn(data)
    resultado = "Vai churnar" if pred == 1 else "Não vai churnar"
    prob_formatada = f"{prob:.2%}"

    with col:
        st.markdown(
            f"""
            <div style="
                background-color: {cor};
                color: #323232;
                width: 225px;
                padding: 10px;
                margin: 10px auto;  
                border-radius: 10px;
                text-align: center;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            ">
                <h2>{plan_name}</h2>
                <p><strong>Previsão:</strong> {resultado}</p>
                <p><strong>Probabilidade de churn:</strong></p>
                <p style="font-size: 32px; font-weight: bold;">{prob_formatada}</p>
                <div style="text-align: left; margin-top: 10px;">
                    <ul>
                        {''.join(f'<li>{item}</li>' for item in description_desc)}
                    </ul>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if st.button(f"Verificar churn - {plan_name}"):
            st.write(input_df.transpose())
            st.success(f"Resultado: {resultado}")
            st.write(f"Probabilidade de churn: {prob_formatada}")
