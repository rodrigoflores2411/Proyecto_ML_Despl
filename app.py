# app.py
import os, json, sys
import numpy as np
import pandas as pd
import joblib
import streamlit as st
from sklearn.base import BaseEstimator, TransformerMixin

st.set_page_config(page_title="Predicción de Riesgo de Diabetes", page_icon="🩺", layout="centered")

# ====== MISMA CLASE QUE EN EL NOTEBOOK ======
class Featurizer(BaseEstimator, TransformerMixin):
    def __init__(self, use_gender=True):
        self.use_gender = use_gender
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        df = pd.DataFrame(X).copy()
        for col in ["PatientID", "BMI_Category"]:
            if col in df.columns:
                df.drop(columns=[col], inplace=True)
        if self.use_gender and "Gender" in df.columns and df["Gender"].dtype == object:
            df["Gender"] = df["Gender"].map({"Female": 0, "Male": 1}).astype(float)
        df["Age_BMI"] = df["Age"] * df["BMI"]
        df["HighBP"]  = (df["BloodPressure"] >= 140).astype(int)
        final_cols = ["Glucose","BloodPressure","Insulin","BMI","Age","DiabetesPedigreeFunction"]
        if self.use_gender: final_cols += ["Gender"]
        final_cols += ["Age_BMI","HighBP"]
        final_cols = [c for c in final_cols if c in df.columns]
        return df[final_cols]

# ====== ALIAS PARA QUE EL PICKLE ENCUENTRE LA CLASE ======
# Muchos notebooks guardan la clase como '__main__.Featurizer' o 'ipykernel_launcher.Featurizer'
current_mod = sys.modules[__name__]
for modname in ("__main__", "ipykernel_launcher", "app"):
    sys.modules.setdefault(modname, current_mod)
    setattr(sys.modules[modname], "Featurizer", Featurizer)

# ====== Carga de artefactos ======
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
    st.error("⚠️ No se encontró **artifacts/model.joblib**. Vuelve a subir el pipeline exportado desde tu notebook.")
    st.stop()

INPUT_FEATURES = meta.get("input_features", [])
USE_GENDER = bool(meta.get("use_gender", True))
GENDER_MAP = {"Female": 0, "Male": 1}

st.title("🩺 Predicción de Riesgo de Diabetes")
st.caption("Pipeline = Featurizer (derivadas) + escalado + Random Forest. La app solo hace inferencia.")
st.write(f"Umbral operativo: **{THRESHOLD:.2f}**")

with st.expander("Campos de entrada esperados"):
    st.code(INPUT_FEATURES)

# ====== UI ======
with st.form("form"):
    c1, c2 = st.columns(2)
    with c1:
        glucose = st.number_input("Glucose (mg/dL)", 0.0, 500.0, 120.0, 1.0)
        bp      = st.number_input("BloodPressure (mmHg)", 0.0, 250.0, 80.0, 1.0)
        insulin = st.number_input("Insulin (µU/mL)", 0.0, 900.0, 85.0, 1.0)
    with c2:
        bmi     = st.number_input("BMI (kg/m²)", 0.0, 80.0, 28.0, 0.1)
        age     = st.number_input("Age (años)", 0, 120, 45, 1)
        dpf     = st.number_input("DiabetesPedigreeFunction", 0.0, 5.0, 0.5, 0.01)
    gender_value = None
    if USE_GENDER and "Gender" in INPUT_FEATURES:
        gender_value = st.selectbox("Gender", list(GENDER_MAP.keys()))
    submitted = st.form_submit_button("Predecir")

def build_raw_row():
    row = {
        "Glucose": glucose, "BloodPressure": bp, "Insulin": insulin,
        "BMI": bmi, "Age": age, "DiabetesPedigreeFunction": dpf,
    }
    if USE_GENDER and "Gender" in INPUT_FEATURES:
        row["Gender"] = GENDER_MAP[gender_value]
    return row

if submitted:
    X_raw = pd.DataFrame([build_raw_row()])
    faltan = [c for c in INPUT_FEATURES if c not in X_raw.columns]
    if faltan:
        st.error(f"Faltan columnas de entrada: {faltan}")
        st.stop()

    if hasattr(pipe, "predict_proba"):
        prob = float(pipe.predict_proba(X_raw)[:, 1][0])
    elif hasattr(pipe, "decision_function"):
        s = float(pipe.decision_function(X_raw)[0])
        prob = 1.0 / (1.0 + np.exp(-s))
    else:
        prob = float(pipe.predict(X_raw)[0])

    pred = int(prob >= THRESHOLD)
    st.subheader("Resultado")
    st.metric("Probabilidad (clase 1)", f"{prob:.2%}")
    st.write(f"**Predicción:** {'Positivo' if pred==1 else 'Negativo'} (umbral {THRESHOLD:.2f})")
    with st.expander("Entrada cruda enviada al Pipeline"):
        st.dataframe(X_raw)
