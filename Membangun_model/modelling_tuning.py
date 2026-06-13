import os
import shutil
import socket
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    RocCurveDisplay,
)
import mlflow
import mlflow.sklearn
import dagshub


def setup_tracking_env():
    """Mengonfigurasi DagsHub online tracking atau local MLflow fallback."""
    dagshub_owner = os.getenv("DAGSHUB_OWNER")
    dagshub_repo = os.getenv("DAGSHUB_REPO")

    if dagshub_owner and dagshub_repo:
        try:
            dagshub.init(repo_owner=dagshub_owner, repo_name=dagshub_repo, mlflow=True)
            print(f"DagsHub terhubung untuk online tracking: {dagshub_owner}/{dagshub_repo}")
        except Exception as e:
            print(f"Koneksi DagsHub gagal: {e}. Menggunakan tracking lokal.")
    else:
        print("Kredensial DagsHub tidak ditemukan di environment. Menggunakan MLflow lokal.")
        a_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        a_socket.settimeout(1.0)
        check = a_socket.connect_ex(("127.0.0.1", 5000))
        a_socket.close()
        if check == 0:
            mlflow.set_tracking_uri("http://127.0.0.1:5000")
            print("Server MLflow lokal terdeteksi aktif pada http://127.0.0.1:5000")
        else:
            print("Server MLflow lokal non-aktif. Menyimpan secara offline di ./mlruns")

    # Eksperimen TERPISAH dari basic, khusus untuk tuning (kriteria submission Skilled)
    mlflow.set_experiment("Diabetes Prediction - Tuning")


def load_and_split_data(filepath):
    """Membaca dataset dan membaginya menjadi data training dan testing."""
    df = pd.read_csv(filepath)
    X = df.drop(columns=["Outcome"])
    y = df["Outcome"]
    return train_test_split(X, y, test_size=0.2, random_state=7, stratify=y)


def perform_grid_search(X_train, y_train):
    """Mencari parameter model Gradient Boosting terbaik menggunakan GridSearchCV."""
    param_grid = {
        'learning_rate': [0.05, 0.1],
        'n_estimators': [50, 100],
        'max_depth': [3, 5]
    }
    gb = GradientBoostingClassifier(random_state=7)
    grid = GridSearchCV(gb, param_grid, cv=3, scoring='f1', n_jobs=-1)
    grid.fit(X_train, y_train)
    return grid.best_estimator_, grid.best_params_


def log_evaluation_metrics(model, X_test, y_test, best_params):
    """Melakukan evaluasi model dan mencatat seluruh parameter, metrik, serta model ke MLflow secara manual (tanpa autolog)."""
    preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds)
    rec = recall_score(y_test, preds)
    f1 = f1_score(y_test, preds)

    print(f"Hasil Evaluasi:\n- Akurasi: {acc:.4f}\n- Presisi: {prec:.4f}\n- Recall: {rec:.4f}\n- F1-Score: {f1:.4f}")

    with mlflow.start_run(run_name="GB_Tuned_Manual"):
        # Log parameter terbaik hasil tuning secara manual
        for param, val in best_params.items():
            mlflow.log_param(param, val)
        mlflow.log_param("cross_validation_kfold", 3)

        # Log metrik performa secara manual
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("f1_score", f1)

        # Simpan model lokal lalu log sebagai artifact folder 'model/'
        temp_model_dir = "temp_model_save"
        if os.path.exists(temp_model_dir):
            shutil.rmtree(temp_model_dir)

        mlflow.sklearn.save_model(model, temp_model_dir)
        mlflow.log_artifacts(temp_model_dir, artifact_path="model")
        shutil.rmtree(temp_model_dir)

        # Artefak 1: Feature Importance Bar Plot
        importances = model.feature_importances_
        feature_names = X_test.columns
        indices = np.argsort(importances)

        plt.figure(figsize=(6, 5))
        plt.barh(range(len(indices)), importances[indices], align='center', color='teal')
        plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
        plt.xlabel('Relative Importance')
        plt.title('Feature Importances - Diabetes')
        plt.tight_layout()

        feat_img = "feature_importance.png"
        plt.savefig(feat_img)
        plt.close()
        mlflow.log_artifact(feat_img)

        # Artefak 2: ROC Curve Plot
        plt.figure(figsize=(6, 5))
        RocCurveDisplay.from_estimator(model, X_test, y_test)
        plt.title('ROC Curve - Diabetes Prediction')
        plt.tight_layout()

        roc_img = "roc_curve.png"
        plt.savefig(roc_img)
        plt.close()
        mlflow.log_artifact(roc_img)

        # Artefak 3: Laporan Klasifikasi (txt)
        report = classification_report(y_test, preds)
        rep_file = "classification_report.txt"
        with open(rep_file, "w") as f:
            f.write(report)
        mlflow.log_artifact(rep_file)

        for temp_file in [feat_img, roc_img, rep_file]:
            if os.path.exists(temp_file):
                os.remove(temp_file)

    print("Seluruh proses tuning dan logging manual untuk Wisesa selesai!")


if __name__ == "__main__":
    setup_tracking_env()
    X_train, X_test, y_train, y_test = load_and_split_data("diabetes_preprocessing.csv")
    best_gb, best_hyperparams = perform_grid_search(X_train, y_train)
    log_evaluation_metrics(best_gb, X_test, y_test, best_hyperparams)
