# app.py
import os
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import streamlit as st

st.set_page_config(page_title="Predicción de Riesgo de Diabetes", page_icon="🩺", layout="centered")

# ========= Carga de artefactos =========
@st.cache_resource
def load_model(path="artifacts/rf_model.joblib"):
    if not os.path.exists(path):
        return None
    try:
        return joblib.load(path)
    except Exception as e:
        st.error(f"No se pudo cargar el modelo desde '{path}': {e}")
        return None

@st.cache_resource
def load_threshold(path="artifacts/threshold.txt", default=0.50):
    try:
        if os.path.exists(path):
            return float(open(path).read().strip())
    except Exception:
        pass
    return float(default)

@st.cache_resource
def load_scaler(path="artifacts/scaler.joblib"):
    # Solo si entrenaste con StandardScaler fuera del pipeline
    return joblib.load(path) if os.path.exists(path) else None

model = load_model()
THRESHOLD = load_threshold()
scaler = load_scaler()

if model is None:
    st.error("⚠️ Falta el modelo en **artifacts/rf_model.joblib**. Expórtalo desde tu notebook.")
    st.stop()

st.title("🩺 Predicción de Riesgo de Diabetes")
st.caption("Modelo final: Random Forest. Esta app realiza **inferencia** con el modelo entrenado en tu notebook.")
st.write(f"Umbral operativo (sensibilidad priorizada): **{THRESHOLD:.2f}**")

# ========= Columnas/orden que espera el modelo =========
DEFAULT_FEATURES = [
    "Glucose", "BloodPressure", "Insulin", "BMI", "Age",
    "DiabetesPedigreeFunction", "Gender",   # quita 'Gender' si no lo usaste
    # Features derivadas que podrías tener en tu notebook:
    "Age_BMI", "HighBP"
]
if hasattr(model, "feature_names_in_"):
    FEATURES = list(model.feature_names_in_)
else:
    FEATURES = DEFAULT_FEATURES

with st.expander("Columnas esperadas por el modelo"):
    st.code(FEATURES)

# Numéricas que escalaste en el notebook (ajusta si cambiaste)
NUMERIC_TO_SCALE = ["Age", "BMI", "BloodPressure", "Insulin", "Glucose", "DiabetesPedigreeFunction"]

GENDER_MAP = {"Female": 0, "Male": 1}

# ========= UI =========
with st.form("form"):
    c1, c2 = st.columns(2)
    with c1:
        glucose = st.number_input("Glucose (mg/dL)", min_value=0.0, max_value=500.0, value=120.0, step=1.0)
        bp      = st.number_input("BloodPressure (mmHg)", min_value=0.0, max_value=250.0, value=80.0, step=1.0)
        insulin = st.number_input("Insulin (µU/mL)", min_value=0.0, max_value=900.0, value=85.0, step=1.0)
    with c2:
        bmi     = st.number_input("BMI (kg/m²)", min_value=0.0, max_value=80.0, value=28.0, step=0.1)
        age     = st.number_input("Age (años)", min_value=0, max_value=120, value=45, step=1)
        dpf     = st.number_input("DiabetesPedigreeFunction", min_value=0.0, max_value=5.0, value=0.5, step=0.01)

    gender_value = None
    if "Gender" in FEATURES:
        gender_value = st.selectbox("Gender", list(GENDER_MAP.keys()))

    submitted = st.form_submit_button("Predecir")

# ========= Construcción de fila y preprocesamiento =========
def build_row_dict():
    row = {
        "Glucose": glucose,
        "BloodPressure": bp,
        "Insulin": insulin,
        "BMI": bmi,
        "Age": age,
        "DiabetesPedigreeFunction": dpf,
    }
    # Features derivadas que usaste en el entrenamiento
    if "Age_BMI" in FEATURES:
        row["Age_BMI"] = age * bmi
    if "HighBP" in FEATURES:
        row["HighBP"] = 1 if bp >= 140 else 0
    if "Gender" in FEATURES:
        row["Gender"] = GENDER_MAP[gender_value]
    return row

def align_and_scale(df: pd.DataFrame) -> pd.DataFrame:
    # Validar faltantes y extras
    faltantes = [c for c in FEATURES if c not in df.columns]
    if faltantes:
        st.error(f"Faltan columnas requeridas por el modelo: {faltantes}")
        st.stop()
    extras = [c for c in df.columns if c not in FEATURES]
    if extras:
        df = df.drop(columns=extras, errors="ignore")
    # Reordenar
    df = df[FEATURES]
    # Escalar SI y solo SI tienes scaler externo
    if scaler is not None:
        cols_to_scale = [c for c in NUMERIC_TO_SCALE if c in df.columns]
        df.loc[:, cols_to_scale] = scaler.transform(df[cols_to_scale])
    return df

# ========= Predicción =========
if submitted:
    X_input = pd.DataFrame([build_row_dict()])
    X_input = align_and_scale(X_input)

    if hasattr(model, "predict_proba"):
        prob = float(model.predict_proba(X_input)[:, 1][0])
    elif hasattr(model, "decision_function"):
        s = float(model.decision_function(X_input)[0])
        prob = 1.0 / (1.0 + np.exp(-s))  # aproximación logística
    else:
        prob = float(model.predict(X_input)[0])  # fallback

    pred = int(prob >= THRESHOLD)

    st.subheader("Resultado")
    st.metric("Probabilidad de riesgo (clase 1)", f"{prob:.2%}")
    st.write(f"**Predicción:** {'Positivo' if pred==1 else 'Negativo'} (umbral {THRESHOLD:.2f})")

    with st.expander("Entrada transformada y alineada"):
        st.dataframe(X_input)
