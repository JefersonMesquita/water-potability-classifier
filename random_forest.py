# Modelo 2: Random Forest - Water Potability Dataset

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
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

# 3. TREINAMENTO DO MODELO
params = {
    "n_estimators": 200,
    "max_depth": None,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "class_weight": "balanced",
    "random_state": 42,
    "n_jobs": -1
}

model = RandomForestClassifier(**params)
model.fit(X_train, y_train)

# 4. AVALIAÇÃO
y_pred = model.predict(X_test)

acc       = accuracy_score(y_test, y_pred)
f1        = f1_score(y_test, y_pred, average="weighted")
precision = precision_score(y_test, y_pred, average="weighted")
recall    = recall_score(y_test, y_pred, average="weighted")

print("\n--- Resultados Random Forest ---")
print(f"Acurácia  : {acc:.4f}")
print(f"F1-Score  : {f1:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print("\nRelatório completo:")
print(classification_report(y_test, y_pred, target_names=["Não potável", "Potável"]))
print("Matriz de Confusão:")
print(confusion_matrix(y_test, y_pred))

# 5. IMPORTÂNCIA DAS FEATURES
print("\n--- Importância das Features ---")
importancias = pd.Series(model.feature_importances_, index=X.columns)
importancias = importancias.sort_values(ascending=False)
for feat, val in importancias.items():
    print(f"  {feat:<20}: {val:.4f}")

# 6. REGISTRO NO MLFLOW
mlflow.set_experiment("water_potability_benchmark")

with mlflow.start_run(run_name="Random_Forest"):
    # Parâmetros
    mlflow.log_params(params)

    # Métricas
    mlflow.log_metric("accuracy",  acc)
    mlflow.log_metric("f1_score",  f1)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall",    recall)

    # Importância das features como dicionário de métricas
    for feat, val in importancias.items():
        mlflow.log_metric(f"feature_importance_{feat}", val)

    # Modelo
    mlflow.sklearn.log_model(model, "random_forest")

    print("\nMLflow: experimento registrado com sucesso!")
    print(f"Acurácia logada: {acc:.4f}")
