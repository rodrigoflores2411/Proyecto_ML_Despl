# app.py
import os, json
import numpy as np
import pandas as pd
import joblib
import streamlit as st

st.set_page_config(page_title="Predicción de Riesgo de Diabetes", page_icon="🩺", layout="centered")

# ========= Carga de artefactos =========
@st.cache_resource
def load_pipeline(path="artifacts/model.joblib"):
    return joblib.load(path) if os.path.exists(path) else None

@st.cache_resource
def load_threshold(path="artifacts/threshold.txt", default=0.50):
    try:
        return float(open(path).read().strip())
    except Exception:
        return float(default)

@st.cache_resource
def load_metadata(path="artifacts/metadata.json"):
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return {"input_features": ["Glucose","BloodPressure","Insulin","BMI","Age","DiabetesPedigreeFunction","Gender"],
            "use_gender": True}

pipe = load_pipeline()
THRESHOLD = load_threshold()
meta = load_metadata()

if pipe is None:
    st.error("⚠️ No se encontró **artifacts/model.joblib**. Sube el pipeline exportado desde tu notebook.")
    st.stop()

INPUT_FEATURES = meta.get("input_features", [])
USE_GENDER = bool(meta.get("use_gender", True))
GENDER_MAP = {"Female": 0, "Male": 1}

st.title("🩺 Predicción de Riesgo de Diabetes")
st.caption("Modelo final empacado como **Pipeline** (preprocesado + Random Forest). La app solo realiza **inferencia**.")
st.write(f"Umbral operativo (sensibilidad priorizada): **{THRESHOLD:.2f}**")

with st.expander("Campos de entrada esperados"):
    st.code(INPUT_FEATURES)

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
    if USE_GENDER and "Gender" in INPUT_FEATURES:
        gender_value = st.selectbox("Gender", list(GENDER_MAP.keys()))

    submitted = st.form_submit_button("Predecir")

# ========= Construcción de la fila bruta (el Pipeline hace el resto) =========
def build_raw_row():
    row = {
        "Glucose": glucose,
        "BloodPressure": bp,
        "Insulin": insulin,
        "BMI": bmi,
        "Age": age,
        "DiabetesPedigreeFunction": dpf,
    }
    if USE_GENDER and "Gender" in INPUT_FEATURES:
        row["Gender"] = GENDER_MAP[gender_value]
    return row

# ========= Predicción =========
if submitted:
    X_raw = pd.DataFrame([build_raw_row()])

    # Validación básica de columnas esperadas por la app
    faltan = [c for c in INPUT_FEATURES if c not in X_raw.columns]
    if faltan:
        st.error(f"Faltan columnas de entrada: {faltan}")
        st.stop()

    # El Pipeline se encarga de: descartar columnas irrelevantes, crear derivadas (Age_BMI, HighBP),
    # estandarizar numéricas y predecir.
    if hasattr(pipe, "predict_proba"):
        prob = float(pipe.predict_proba(X_raw)[:, 1][0])
    elif hasattr(pipe, "decision_function"):
        s = float(pipe.decision_function(X_raw)[0])
        prob = 1.0 / (1.0 + np.exp(-s))
    else:
        prob = float(pipe.predict(X_raw)[0])

    pred = int(prob >= THRESHOLD)

    st.subheader("Resultado")
    st.metric("Probabilidad de riesgo (clase 1)", f"{prob:.2%}")
    st.write(f"**Predicción:** {'Positivo' if pred==1 else 'Negativo'} (umbral {THRESHOLD:.2f})")

    with st.expander("Entrada cruda enviada al Pipeline"):
        st.dataframe(X_raw)
