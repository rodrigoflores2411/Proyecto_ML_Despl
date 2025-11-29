
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class Featurizer(BaseEstimator, TransformerMixin):
    """
    Transformador custom del proyecto.
    - Limpia columnas irrelevantes si aparecen
    - Mapea Gender (Female/Male -> 0/1) si se usa
    - Crea derivadas: Age_BMI, HighBP
    - Devuelve columnas en el orden esperado por el preprocesador
    """
    def __init__(self, use_gender: bool = True):
        self.use_gender = use_gender

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = pd.DataFrame(X).copy()

        # Quita columnas si vinieran en el DataFrame
        for col in ["PatientID", "BMI_Category"]:
            if col in df.columns:
                df.drop(columns=[col], inplace=True)

        # Map de género si viene como texto
        if self.use_gender and "Gender" in df.columns and df["Gender"].dtype == object:
            df["Gender"] = df["Gender"].map({"Female": 0, "Male": 1}).astype(float)

        # Derivadas
        df["Age_BMI"] = df["Age"] * df["BMI"]
        df["HighBP"]  = (df["BloodPressure"] >= 140).astype(int)

        # Orden de columnas final
        final_cols = ["Glucose","BloodPressure","Insulin","BMI","Age","DiabetesPedigreeFunction"]
        if self.use_gender:
            final_cols += ["Gender"]
        final_cols += ["Age_BMI","HighBP"]
        final_cols = [c for c in final_cols if c in df.columns]
        return df[final_cols]
