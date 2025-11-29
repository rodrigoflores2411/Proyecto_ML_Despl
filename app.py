
# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

st.set_page_config(page_title="Diabetes Risk Predictor", page_icon="🩺", layout="centered")

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
def load_threshold(path="artifacts/threshold.txt", default=0.5):
    try:
        if os.path.exists(path):
            return float(open(path).read().strip())
    except Exception:
        pass
    return float(default)

model = load_model()
THRESHOLD = load_threshold()

st.title("🩺 Predicción de Riesgo de Diabetes")
st.caption("Modelo final: Random Forest. Esta app solo realiza inferencia.")

DEFAULT_FEATURES = [
    "Glucose",
    "BloodPressure",
    "Insulin",
    "BMI",
    "Age",
    "DiabetesPedigreeFunction",
    "Gender"
]

def guess_features_from_model(m):
    feats = None
    try:
        feats = list(m.feature_names_in_)
    except Exception:
        try:
            last = getattr(m, "steps", [])[-1][1]
            feats = list(getattr(last, "feature_names_in_", []))
        except Exception:
            feats = None
    return feats if feats else DEFAULT_FEATURES

FEATURES = guess_features_from_model(model) if model is not None else DEFAULT_FEATURES
GENDER_MAP = {"Female": 0, "Male": 1}

if model is None:
    st.error("⚠️ Falta 'artifacts/rf_model.joblib'. Sube el modelo exportado desde el notebook.")
    st.info("También puedes ajustar 'artifacts/threshold.txt' con tu umbral operativo (por defecto 0.50).")
    st.stop()

st.write(f"Umbral operativo (sensibilidad priorizada): **{THRESHOLD:.2f}**")
with st.expander("Columnas esperadas"):
    st.code(str(FEATURES))

with st.form("form"):
    col1, col2 = st.columns(2)
    with col1:
        glucose = st.number_input("Glucose (mg/dL)", 0.0, 500.0, 120.0, 1.0)
        bp      = st.number_input("BloodPressure (mmHg)", 0.0, 250.0, 80.0, 1.0)
        insulin = st.number_input("Insulin (µU/mL)", 0.0, 900.0, 85.0, 1.0)
    with col2:
        bmi     = st.number_input("BMI (kg/m²)", 0.0, 80.0, 28.0, 0.1)
        age     = st.number_input("Age (años)", 0, 120, 45, 1)
        dpf     = st.number_input("DiabetesPedigreeFunction", 0.0, 5.0, 0.5, 0.01)

    gender_value = None
    if "Gender" in FEATURES:
        gender_value = st.selectbox("Gender", ["Female", "Male"])

    submitted = st.form_submit_button("Predecir")

def build_row():
    row = {
        "Glucose": glucose,
        "BloodPressure": bp,
        "Insulin": insulin,
        "BMI": bmi,
        "Age": age,
        "DiabetesPedigreeFunction": dpf,
    }
    if "Gender" in FEATURES:
        row["Gender"] = {"Female": 0, "Male": 1}[gender_value]
    return row

if submitted:
    X_input = pd.DataFrame([build_row()])
    for c in FEATURES:
        if c not in X_input.columns:
            X_input[c] = 0
    X_input = X_input[FEATURES]

    if hasattr(model, "predict_proba"):
        prob = float(model.predict_proba(X_input)[:, 1][0])
    elif hasattr(model, "decision_function"):
        s = float(model.decision_function(X_input)[0])
        prob = 1.0 / (1.0 + np.exp(-s))
    else:
        pred = int(model.predict(X_input)[0])
        prob = float(pred)

    pred_class = int(prob >= THRESHOLD)

    st.subheader("Resultado")
    st.metric("Probabilidad de riesgo (clase 1)", f"{prob:.2%}")
    st.write(f"**Predicción:** {'Positivo' if pred_class==1 else 'Negativo'}  (umbral {THRESHOLD:.2f})")

    with st.expander("Entrada formateada"):
        st.dataframe(X_input)
