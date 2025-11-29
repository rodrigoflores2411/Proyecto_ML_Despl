# Proyecto_ML_Despl

# Diabetes Risk Predictor (Streamlit)

App de **inferencia** para el proyecto de Machine Learning. El entrenamiento se realiza en el **notebook**;
aquí solo se **carga el modelo** y se predice a partir de inputs del usuario.

## Estructura
```
repo/
├─ app.py
├─ artifacts/
│   ├─ rf_model.joblib       # (lo exportas desde tu notebook)
│   └─ threshold.txt         # (umbral operativo, p.ej. 0.47)
├─ requirements.txt
└─ README.md
```

## 1) Exporta artefactos desde tu notebook
```python
import joblib, os
os.makedirs("artifacts", exist_ok=True)

# Guarda el modelo o pipeline final
joblib.dump(rf_model, "artifacts/rf_model.joblib")

# Guarda el umbral operativo (thr_opt que calculaste con la curva ROC)
with open("artifacts/threshold.txt", "w") as f:
    f.write(str(thr_opt))  # por ejemplo, 0.47
```
> Si aún no tienes `thr_opt`, usa 0.50 y ajústalo luego.

## 2) Ejecuta localmente
```bash
pip install -r requirements.txt
streamlit run app.py
```

## 3) Despliegue (Streamlit Community Cloud)
1. Sube este repo a GitHub.
2. Crea **New app** y apunta a `app.py`.
3. Asegúrate de incluir `artifacts/rf_model.joblib` y `artifacts/threshold.txt`.
   Si faltan, la app te lo indicará.
