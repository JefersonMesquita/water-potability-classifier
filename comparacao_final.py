# Comparação Final - Benchmark dos 3 Modelos - Water Potability Dataset

import mlflow
import pandas as pd

# 1. BUSCAR RUNS DO EXPERIMENTO NO MLFLOW
mlflow.set_experiment("water_potability_benchmark")

runs = mlflow.search_runs(
    experiment_names=["water_potability_benchmark"],
    order_by=["metrics.accuracy DESC"]
)

if runs.empty:
    print("Nenhum run encontrado. Execute os scripts dos 3 modelos primeiro.")
    exit()

# Manter apenas o run mais recente de cada modelo (caso tenha rodado mais de uma vez)
runs = runs.sort_values("start_time", ascending=False)
runs = runs.drop_duplicates(subset=["tags.mlflow.runName"], keep="first")
runs = runs.sort_values("metrics.accuracy", ascending=False)

# 2. MONTAR TABELA COMPARATIVA
colunas = {
    "tags.mlflow.runName": "Modelo",
    "metrics.accuracy":    "Acurácia",
    "metrics.f1_score":    "F1-Score",
    "metrics.precision":   "Precision",
    "metrics.recall":      "Recall",
}

df_resultado = runs[list(colunas.keys())].rename(columns=colunas).reset_index(drop=True)

# Formatar como percentual
for col in ["Acurácia", "F1-Score", "Precision", "Recall"]:
    df_resultado[col] = df_resultado[col].apply(lambda x: f"{x*100:.2f}%")

print("\n" + "="*65)
print("   BENCHMARK FINAL - COMPARAÇÃO DOS 3 MODELOS")
print("   Dataset: Water Potability | Teste: 20% estratificado")
print("="*65)
print(df_resultado.to_string(index=False))
print("="*65)

# 3. RANKING E MELHOR MODELO
runs_ordenados = runs.sort_values("metrics.accuracy", ascending=False)
melhor = runs_ordenados.iloc[0]

print(f"\n🏆 Melhor modelo: {melhor['tags.mlflow.runName']}")
print(f"   Acurácia : {melhor['metrics.accuracy']*100:.2f}%")
print(f"   F1-Score : {melhor['metrics.f1_score']*100:.2f}%")
print(f"   Precision: {melhor['metrics.precision']*100:.2f}%")
print(f"   Recall   : {melhor['metrics.recall']*100:.2f}%")
print("\nTodos os experimentos disponíveis no MLflow UI:")
print("   Execute: mlflow ui")
print("   Acesse : http://localhost:5000")