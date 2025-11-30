import streamlit as st
import pandas as pd
import joblib

# ---------- Cargar modelo y preprocesado ----------
@st.cache_resource
def load_artifacts():
    # El archivo debe estar en el mismo repo que este app.py
    artifacts = joblib.load("diabetes_rf_model.joblib")
    return artifacts

artifacts = load_artifacts()
model = artifacts["model"]
scaler = artifacts["scaler"]
num_cols = artifacts["num_cols"]
feature_cols = artifacts["feature_cols"]

# ---------- Configuración de la página ----------
st.set_page_config(
    page_title="Predicción de Riesgo de Diabetes",
    page_icon="🩺",
    layout="centered"
)

st.title("🩺 Predicción de Riesgo de Diabetes")
st.write(
    """
    Esta aplicación utiliza un modelo de **Machine Learning (Random Forest)** entrenado
    para estimar la **probabilidad de diabetes** a partir de indicadores de salud básicos.

    Ingresa los datos del paciente y presiona **"Calcular riesgo"** para obtener la
    estimación del modelo.
    """
)

# ---------- Función para construir las features ----------
def build_features(age, gender, bmi, bp, glucose, insulin, dpf):
    # Mismo encoding que en el notebook
    gender_num = 1 if gender == "Female" else 0
    high_bp = 1 if bp >= 140 else 0
    age_bmi = age * bmi

    data = {
        "Age": age,
        "BMI": bmi,
        "BloodPressure": bp,
        "Insulin": insulin,
        "Glucose": glucose,
        "DiabetesPedigreeFunction": dpf,
        "Gender": gender_num,
        "Age_BMI": age_bmi,
        "HighBP": high_bp,
    }

    df = pd.DataFrame([data])

    # Ordenar columnas igual que en el entrenamiento
    df = df.reindex(columns=feature_cols, fill_value=0)

    # Escalar solo columnas numéricas originales
    df[num_cols] = scaler.transform(df[num_cols])

    return df

# ---------- Formulario de entrada ----------
with st.form("form_diabetes"):
    st.subheader("Datos del paciente")

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Edad (años)", min_value=1, max_value=120, value=40)
        gender = st.radio("Género", ["Male", "Female"], index=0)
        bmi = st.number_input(
            "Índice de Masa Corporal (BMI)",
            min_value=10.0, max_value=70.0, value=25.0, step=0.1
        )

    with col2:
        bp = st.number_input(
            "Presión arterial sistólica (mmHg)",
            min_value=40.0, max_value=250.0, value=120.0, step=1.0
        )
        glucose = st.number_input(
            "Glucosa (mg/dL)",
            min_value=0.0, max_value=300.0, value=120.0, step=1.0
        )
        insulin = st.number_input(
            "Insulina",
            min_value=0.0, max_value=400.0, value=80.0, step=1.0
        )

    dpf = st.number_input(
        "Diabetes Pedigree Function (historial familiar)",
        min_value=0.0, max_value=3.0, value=0.5, step=0.01
    )

    submitted = st.form_submit_button("Calcular riesgo")

# ---------- Predicción ----------
if submitted:
    X_new = build_features(age, gender, bmi, bp, glucose, insulin, dpf)

    prob = float(model.predict_proba(X_new)[0, 1])
    pred = int(model.predict(X_new)[0])

    st.subheader("Resultado")
    st.metric("Probabilidad estimada de diabetes", f"{prob*100:.1f} %")

    if pred == 1:
        st.error("Clasificación del modelo: **ALTO RIESGO / POSITIVO**")
    else:
        st.success("Clasificación del modelo: **BAJO RIESGO / NEGATIVO**")

    st.caption(
        "⚠️ Este resultado es solo orientativo y **no reemplaza** una evaluación médica profesional."
    )
