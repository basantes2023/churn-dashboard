# =========================================
# 🚀 CNT EP - DASHBOARD PRO 10/10
# =========================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

st.set_page_config(page_title="CNT EP Churn PRO", layout="wide")

# ==============================
# CARGAR DATOS
# ==============================
@st.cache_data
def load_data():
    df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df = df.dropna()
    return df

df = load_data()

# ==============================
# MODELO
# ==============================
model = joblib.load("modelo_churn.pkl")
columns = joblib.load("columnas.pkl")

# ==============================
# TITULO
# ==============================
st.title("🚀 CNT EP - Sistema Inteligente Anti-Churn")
st.markdown("### 📊 Analítica avanzada + IA para retención de clientes")

# ==============================
# KPIs
# ==============================
total_clientes = len(df)
churn_rate = (df["Churn"] == "Yes").mean() * 100
ingreso_prom = df["MonthlyCharges"].mean()

col1, col2, col3, col4 = st.columns(4)

col1.metric("👥 Clientes", total_clientes)
col2.metric("📉 Churn Rate", f"{churn_rate:.2f}%")
col3.metric("💰 Ingreso Promedio", f"${ingreso_prom:.2f}")
col4.metric("🎯 Meta", "< 20%")

st.markdown("---")

# ==============================
# FILTROS
# ==============================
st.sidebar.header("🔎 Filtros")

contract = st.sidebar.multiselect(
    "Contrato",
    df["Contract"].unique(),
    default=df["Contract"].unique()
)

internet = st.sidebar.multiselect(
    "Internet",
    df["InternetService"].unique(),
    default=df["InternetService"].unique()
)

df_f = df[
    (df["Contract"].isin(contract)) &
    (df["InternetService"].isin(internet))
]

# ==============================
# SEGMENTACIÓN
# ==============================
st.subheader("🎯 Segmentación de Clientes")

alto = int(len(df_f) * 0.26)
medio = int(len(df_f) * 0.20)
bajo = int(len(df_f) * 0.54)

c1, c2, c3 = st.columns(3)

c1.error(f"🔴 Alto riesgo: {alto}")
c2.warning(f"🟡 Riesgo medio: {medio}")
c3.success(f"🟢 Bajo riesgo: {bajo}")

# ==============================
# GRAFICOS
# ==============================
st.subheader("📊 Análisis")

col1, col2 = st.columns(2)

with col1:
    fig, ax = plt.subplots()
    sns.countplot(x='Churn', data=df_f, ax=ax)
    st.pyplot(fig)

with col2:
    fig, ax = plt.subplots()
    sns.countplot(x='Contract', hue='Churn', data=df_f, ax=ax)
    plt.xticks(rotation=45)
    st.pyplot(fig)

# ==============================
# IMPACTO ECONÓMICO
# ==============================
st.subheader("💰 Impacto Económico")

ahorro = (churn_rate - 20) * total_clientes * ingreso_prom * 0.1

st.info(f"Reducir churn a 20% podría generar ahorro estimado de: ${ahorro:,.2f}")

# ==============================
# BUYER PERSONA
# ==============================
st.subheader("👤 Buyer Persona")

col1, col2 = st.columns(2)

with col1:
    st.error("""
    🔴 María Palacios  
    - Alta probabilidad de churn  
    - Contrato mensual  
    - Alta sensibilidad a precio  
    """)

with col2:
    st.success("""
    🟢 Carlos Farías  
    - Cliente leal  
    - Paquete completo  
    - Alta estabilidad  
    """)

# ==============================
# PREDICCIÓN
# ==============================
st.subheader("🤖 Predicción de Churn")

tenure = st.slider("Antigüedad", 0, 72, 12)
monthly = st.slider("Pago mensual", 10, 120, 50)
total = tenure * monthly

input_data = pd.DataFrame([[0]*len(columns)], columns=columns)

if "tenure" in input_data:
    input_data["tenure"] = tenure

if "MonthlyCharges" in input_data:
    input_data["MonthlyCharges"] = monthly

if "TotalCharges" in input_data:
    input_data["TotalCharges"] = total

if st.button("🔮 Predecir"):
    pred = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0][1]

    if pred == 1:
        st.error(f"🔴 Riesgo ALTO ({prob*100:.2f}%)")
        st.warning("📢 Recomendación: ofrecer descuento + soporte prioritario")
    else:
        st.success(f"🟢 Cliente estable ({prob*100:.2f}%)")
        st.info("💡 Recomendación: ofrecer upgrade o fidelización")

# ==============================
# TABLA
# ==============================
st.subheader("📋 Datos")
st.dataframe(df_f)
