# === Features desde el modelo (obligatorio si existen) ===
DEFAULT_FEATURES = [
    "Glucose","BloodPressure","Insulin","BMI","Age","DiabetesPedigreeFunction","Gender"
]
if hasattr(model, "feature_names_in_"):
    FEATURES = list(model.feature_names_in_)
else:
    FEATURES = DEFAULT_FEATURES

# ... (widgets de entrada iguales a los que ya tienes) ...

if submitted:
    # Construir fila
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

    X_input = pd.DataFrame([row])

    # Validar columnas exactas
    faltantes = [c for c in FEATURES if c not in X_input.columns]
    extras    = [c for c in X_input.columns if c not in FEATURES]
    if faltantes:
        st.error(f"Faltan columnas requeridas por el modelo: {faltantes}. "
                 "Ajusta los inputs/FEATURES para coincidir con el entrenamiento.")
        st.stop()
    if extras:
        # elimina columnas que el modelo no conoce
        X_input = X_input.drop(columns=extras, errors="ignore")

    # Reordenar exactamente como espera el modelo
    X_input = X_input[FEATURES]

    # ==== Predicción ====
    if hasattr(model, "predict_proba"):
        prob = float(model.predict_proba(X_input)[:, 1][0])
    elif hasattr(model, "decision_function"):
        s = float(model.decision_function(X_input)[0])
        prob = 1 / (1 + np.exp(-s))
    else:
        prob = float(model.predict(X_input)[0])

    pred_class = int(prob >= THRESHOLD)

    st.subheader("Resultado")
    st.metric("Probabilidad de riesgo (clase 1)", f"{prob:.2%}")
    st.write(f"**Predicción:** {'Positivo' if pred_class==1 else 'Negativo'} (umbral {THRESHOLD:.2f})")
    with st.expander("Ver entrada formateada"):
        st.dataframe(X_input)
