import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
import mlflow
import mlflow.sklearn

def main():
    # Set nama eksperimen di MLflow
    mlflow.set_experiment("Diabetes Prediction - wisesa_sutresna")
    
    # Aktifkan pencatatan otomatis (autolog)
    mlflow.sklearn.autolog()
    
    print("Memuat dataset hasil preprocessing...")
    data = pd.read_csv("diabetes_preprocessing.csv")
    
    # Bagi fitur dan target (target: Outcome)
    X = data.drop(columns=["Outcome"])
    y = data["Outcome"]
    
    # Lakukan pembagian training dan testing set
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=7, stratify=y
    )
    
    print("Memulai pencatatan MLflow untuk baseline model...")
    with mlflow.start_run(run_name="GB_Baseline"):
        # Buat model default Gradient Boosting
        model = GradientBoostingClassifier(random_state=7)
        model.fit(X_train, y_train)
        
        # Prediksi dan evaluasi
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"Akurasi Baseline Gradient Boosting: {acc:.4f}")
        print("\nLaporan Klasifikasi Baseline:")
        print(classification_report(y_test, y_pred))

if __name__ == "__main__":
    main()
