# Modelo 1: Regressão Logística

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, classification_report, confusion_matrix
)

# 1. CARREGAMENTO DO DATASET TRATADO
INPUT_PATH = "water_potability_tratado.csv"

df = pd.read_csv(INPUT_PATH)
print(f"Dataset carregado: {df.shape[0]} linhas, {df.shape[1]} colunas")

X = df.drop(columns=["Potability"])
y = df["Potability"]

# 2. SPLIT ESTRATIFICADO (80% treino, 20% teste)
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f"\nTreino: {X_train.shape[0]} amostras | Teste: {X_test.shape[0]} amostras")
print(f"Proporção treino - Não potável: {(y_train==0).sum()} | Potável: {(y_train==1).sum()}")
print(f"Proporção teste  - Não potável: {(y_test==0).sum()}  | Potável: {(y_test==1).sum()}")

# 3. NORMALIZAÇÃO COM STANDARDSCALER
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# 4. TREINAMENTO DO MODELO
params = {
    "C": 1.0,
    "max_iter": 1000,
    "class_weight": "balanced",
    "solver": "lbfgs",
    "random_state": 42
}

model = LogisticRegression(**params)
model.fit(X_train_scaled, y_train)

# 5. AVALIAÇÃO
y_pred = model.predict(X_test_scaled)

acc       = accuracy_score(y_test, y_pred)
f1        = f1_score(y_test, y_pred, average="weighted")
precision = precision_score(y_test, y_pred, average="weighted")
recall    = recall_score(y_test, y_pred, average="weighted")

print("\n--- Resultados Regressão Logística ---")
print(f"Acurácia  : {acc:.4f}")
print(f"F1-Score  : {f1:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print("\nRelatório completo:")
print(classification_report(y_test, y_pred, target_names=["Não potável", "Potável"]))
print("Matriz de Confusão:")
print(confusion_matrix(y_test, y_pred))

# 6. REGISTRO NO MLFLOW
mlflow.set_experiment("water_potability_benchmark")

with mlflow.start_run(run_name="Regressao_Logistica"):
    # Parâmetros
    mlflow.log_params(params)

    # Métricas
    mlflow.log_metric("accuracy",  acc)
    mlflow.log_metric("f1_score",  f1)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall",    recall)

    # Modelo
    mlflow.sklearn.log_model(model, "regressao_logistica")

    print("\nMLflow: experimento registrado com sucesso!")
    print(f"Acurácia logada: {acc:.4f}")
