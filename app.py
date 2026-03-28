# =========================================
# APP STREAMLIT - CHURN CNT EP (VERSIÓN PRO)
# =========================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# CONFIGURACIÓN
st.set_page_config(
    page_title="CNT EP - Dashboard Churn",
    layout="wide"
)

# =========================================
# CARGAR DATOS
# =========================================
@st.cache_data
def load_data():
    df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df = df.dropna()
    return df

df = load_data()

# =========================================
# CARGAR MODELO
# =========================================
model = joblib.load("modelo_churn.pkl")
columns = joblib.load("columnas.pkl")

# =========================================
# TITULO
# =========================================
st.title("📊 CNT EP - Dashboard Inteligente de Churn")
st.markdown("Análisis y predicción de deserción de clientes")

# =========================================
# SIDEBAR FILTROS
# =========================================
st.sidebar.header("🔎 Filtros")

contract_filter = st.sidebar.multiselect(
    "Tipo de contrato",
    options=df["Contract"].unique(),
    default=df["Contract"].unique()
)

internet_filter = st.sidebar.multiselect(
    "Tipo de internet",
    options=df["InternetService"].unique(),
    default=df["InternetService"].unique()
)

df_filtered = df[
    (df["Contract"].isin(contract_filter)) &
    (df["InternetService"].isin(internet_filter))
]

# =========================================
# KPIs
# =========================================
col1, col2, col3 = st.columns(3)

churn_rate = (df_filtered["Churn"] == "Yes").mean() * 100

col1.metric("Clientes", len(df_filtered))
col2.metric("Churn %", f"{churn_rate:.2f}%")
col3.metric("Ingreso Promedio", f"${df_filtered['MonthlyCharges'].mean():.2f}")

st.markdown("---")

# =========================================
# GRAFICOS
# =========================================
st.subheader("📊 Análisis de Churn")

col1, col2 = st.columns(2)

# Gráfico 1
with col1:
    fig, ax = plt.subplots()
    sns.countplot(x='Churn', data=df_filtered, ax=ax)
    ax.set_title("Distribución de Churn")
    st.pyplot(fig)

# Gráfico 2
with col2:
    fig, ax = plt.subplots()
    sns.countplot(x='Contract', hue='Churn', data=df_filtered, ax=ax)
    plt.xticks(rotation=45)
    ax.set_title("Churn por tipo de contrato")
    st.pyplot(fig)

# =========================================
# CORRELACIÓN
# =========================================
st.subheader("📉 Correlación")

df_numeric = df_filtered.select_dtypes(include=['number'])

fig, ax = plt.subplots(figsize=(10,6))
sns.heatmap(df_numeric.corr(), annot=True, cmap="coolwarm", ax=ax)
st.pyplot(fig)

# =========================================
# IMPORTANCIA DE VARIABLES
# =========================================
st.subheader("🔥 Variables más importantes")

try:
    importances = model.feature_importances_
    feat_imp = pd.Series(importances, index=columns).sort_values(ascending=False)

    fig, ax = plt.subplots()
    feat_imp.head(10).plot(kind='bar', ax=ax)
    ax.set_title("Top Variables - Random Forest")
    st.pyplot(fig)
except:
    st.warning("El modelo no soporta importancia de variables")

# =========================================
# PREDICCIÓN
# =========================================
st.markdown("---")
st.subheader("🤖 Predicción de Churn")

st.sidebar.header("📌 Datos del cliente")

tenure = st.sidebar.slider("Antigüedad (meses)", 0, 72, 12)
monthly = st.sidebar.slider("Pago mensual", 10, 120, 50)
total = tenure * monthly

# Crear vector base
input_data = pd.DataFrame([[0]*len(columns)], columns=columns)

# Asignar valores
if "tenure" in input_data:
    input_data["tenure"] = tenure

if "MonthlyCharges" in input_data:
    input_data["MonthlyCharges"] = monthly

if "TotalCharges" in input_data:
    input_data["TotalCharges"] = total

# Botón predicción
if st.sidebar.button("🔮 Predecir"):
    pred = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0][1]

    st.subheader("Resultado")

    if pred == 1:
        st.error(f"🔴 Alto riesgo de churn ({prob*100:.2f}%)")
    else:
        st.success(f"🟢 Cliente estable ({prob*100:.2f}%)")

# =========================================
# TABLA
# =========================================
st.markdown("---")
st.subheader("📋 Datos filtrados")
st.dataframe(df_filtered)