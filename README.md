# Classificador de Potabilidade da Água

Projeto de automação na análise de dados da crise global da água potável 

---

## Apresentação em Vídeo

> **Substitua o link abaixo pelo vídeo não-listado no YouTube (máx. 10 minutos).**

[![Apresentação do Projeto](https://img.shields.io/badge/YouTube-Apresentação-red?logo=youtube)](https://www.youtube.com/watch?v=SEU_VIDEO_AQUI)

**Link:** https://www.youtube.com/watch?v=0qtMGzikgFo

---

## Equipe

| Integrante | Matrícula |
|------------|-----------|
| José Ericson Silveira Teófilo | 508057 |
| Igor da Silva Pierre | 470562 |
| Jeferson Rodrigo Silva de Mesquita | 511424 |
| Antonio Lucas Damasceno Melo | 514783 |

---

## Sobre o Projeto

De acordo com a Organização Mundial da Saúde (OMS) e a UNICEF, bilhões de pessoas sofrem com a falta de acesso à água segura para consumo. Este projeto desenvolve e avalia modelos preditivos de **classificação supervisionada** para determinar se uma amostra de água é **potável** (`1`) ou **não potável** (`0`), servindo como ferramenta de triagem automatizada e de baixo custo para populações vulneráveis.

A metodologia abrange:

- Coleta e tratamento dos dados
- Análise estatística da base
- Treinamento e avaliação de **três algoritmos** de aprendizado supervisionado
- Comunicação dos resultados via **MLflow**

---

## Dataset

**Fonte:** [Water Potability — Kaggle](https://www.kaggle.com/datasets/adityakadiwal/water-potability)

O dataset original apresenta desafios comuns em problemas reais de ciência de dados: **desbalanceamento de classes** e **valores ausentes**. Para garantir a reprodutibilidade do projeto, o repositório inclui o arquivo já tratado `water_potability_tratado.csv` (3.276 amostras).

### Atributos

| Atributo | Descrição | Unidade |
|----------|-----------|---------|
| `ph` | Equilíbrio ácido-base da água (OMS recomenda 6,5–8,5) | escala 0–14 |
| `Hardness` | Dureza causada por sais de cálcio e magnésio | mg/L |
| `Solids` | Sólidos dissolvidos totais (TDS) | ppm |
| `Chloramines` | Concentração de cloraminas (desinfetantes) | ppm |
| `Sulfate` | Concentração de sulfatos | mg/L |
| `Conductivity` | Condutividade elétrica | μS/cm |
| `Organic_carbon` | Carbono orgânico total (TOC) | ppm |
| `Trihalomethanes` | Subprodutos químicos do tratamento da água | μg/L |
| `Turbidity` | Turbidez (matéria suspensa) | NTU |
| `Potability` | **Alvo:** 1 = potável, 0 = não potável | — |

---

## Metodologia

### Pré-processamento

O dataset original foi tratado para lidar com valores ausentes e inconsistências. O arquivo resultante (`water_potability_tratado.csv`) é utilizado por todos os scripts de treinamento, garantindo comparação justa entre os modelos.

### Divisão dos dados

- **Treino:** 80% das amostras  
- **Teste:** 20% das amostras  
- **Estratificação:** mantida a proporção das classes em ambos os conjuntos  
- **Semente aleatória:** `random_state=42` (reprodutibilidade)

### Tratamento do desbalanceamento

| Modelo | Estratégia |
|--------|------------|
| Regressão Logística | `class_weight="balanced"` + normalização com `StandardScaler` |
| Random Forest | `class_weight="balanced"` |
| XGBoost | `scale_pos_weight` calculado como razão negativos/positivos no treino |

### Métricas de avaliação

Todos os modelos são avaliados com:

- **Acurácia**
- **F1-Score** (média ponderada)
- **Precision** (média ponderada)
- **Recall** (média ponderada)
- Matriz de confusão e relatório de classificação

---

## Modelos Implementados

| Script | Algoritmo | Principais hiperparâmetros |
|--------|-----------|----------------------------|
| `regressao_logistica.py` | Regressão Logística | `C=1.0`, `solver="lbfgs"`, `class_weight="balanced"` |
| `random_forest.py` | Random Forest | `n_estimators=200`, `class_weight="balanced"` |
| `model_xgboost.py` | XGBoost | `n_estimators=200`, `max_depth=6`, `learning_rate=0.1` |

Cada script registra automaticamente parâmetros, métricas e o modelo treinado no experimento MLflow `water_potability_benchmark`.

---

## MLflow

O [MLflow](https://mlflow.org/) é utilizado para rastrear e comparar os experimentos de forma centralizada.

### Visualizar os experimentos

Após executar os três modelos, inicie a interface do MLflow:

```bash
mlflow ui
```

Acesse no navegador: **http://localhost:5000**

### Comparar os modelos

O script `comparacao_final.py` consulta os runs registrados, monta uma tabela comparativa e indica o melhor modelo com base na acurácia:

```bash
python comparacao_final.py
```

---

## Estrutura do Repositório

```
water-potability-classifier/
├── water_potability_tratado.csv   # Dataset pré-processado
├── regressao_logistica.py         # Modelo 1 — Regressão Logística
├── random_forest.py               # Modelo 2 — Random Forest
├── model_xgboost.py               # Modelo 3 — XGBoost
├── comparacao_final.py            # Benchmark comparativo via MLflow
├── requirements.txt               # Dependências do projeto
└── README.md                      # Este arquivo
```

---

## Requisitos

- **Python** 3.10 ou superior
- **pip** (gerenciador de pacotes)

### Dependências principais

| Pacote | Versão |
|--------|--------|
| scikit-learn | 1.8.0 |
| xgboost | 3.3.0 |
| mlflow | 3.14.0 |
| pandas | 2.3.3 |
| numpy | 2.4.4 |

> O desbalanceamento das classes é tratado diretamente nos modelos (`class_weight` e `scale_pos_weight`), sem uso de técnicas de reamostragem como SMOTE.

---

## Instalação

### 1. Clonar o repositório

```bash
git clone https://github.com/JefersonMesquita/water-potability-classifier.git
cd water-potability-classifier
```

### 2. Criar e ativar um ambiente virtual (recomendado)

```bash
python -m venv venv

# Linux / macOS
source venv/bin/activate

# Windows (PowerShell)
.\venv\Scripts\Activate.ps1
```

### 3. Instalar as dependências

```bash
pip install -r requirements.txt
```

---

## Execução

Execute os scripts na ordem abaixo, a partir da raiz do projeto:

```bash
# 1. Treinar Regressão Logística
python regressao_logistica.py

# 2. Treinar Random Forest
python random_forest.py

# 3. Treinar XGBoost
python model_xgboost.py

# 4. Comparar os resultados
python comparacao_final.py

# 5. (Opcional) Abrir interface do MLflow
mlflow ui
```

Cada script imprime no terminal as métricas de avaliação, a matriz de confusão e a confirmação do registro no MLflow. Os modelos Random Forest e XGBoost também exibem a importância das features.

---

## Resultados

Benchmark obtido com split 80/20 estratificado (`random_state=42`), conjunto de teste com 656 amostras:

| Modelo | Acurácia | F1-Score | Precision | Recall |
|--------|----------|----------|-----------|--------|
| **Random Forest** | **78,96%** | 78,11% | 79,44% | 78,96% |
| XGBoost | 78,20% | 78,01% | 77,99% | 78,20% |
| Regressão Logística | 54,57% | 55,13% | 56,91% | 54,57% |

**Melhor modelo:** Random Forest

Saída do `comparacao_final.py`:

```
=================================================================
   BENCHMARK FINAL - COMPARAÇÃO DOS 3 MODELOS
   Dataset: Water Potability | Teste: 20% estratificado
=================================================================
             Modelo Acurácia F1-Score Precision Recall
      Random_Forest   78.96%   78.11%    79.44% 78.96%
            XGBoost   78.20%   78.01%    77.99% 78.20%
Regressao_Logistica   54.57%   55.13%    56.91% 54.57%
=================================================================

🏆 Melhor modelo: Random_Forest
```

**Observações:** Random Forest e XGBoost tiveram desempenho semelhante (~78%). A Regressão Logística ficou próxima do acaso (~55%), servindo como baseline. Em ambos os ensembles, as features mais relevantes foram **Sulfate** e **ph**.

---

## Referências

- Organização Mundial da Saúde (OMS) e UNICEF — acesso à água potável
- [Water Potability Dataset — Kaggle](https://www.kaggle.com/datasets/adityakadiwal/water-potability)
- [MLflow — Machine Learning Lifecycle Platform](https://mlflow.org/)
- Scikit-learn Documentation — [https://scikit-learn.org/](https://scikit-learn.org/)
- XGBoost Documentation — [https://xgboost.readthedocs.io/](https://xgboost.readthedocs.io/)

---

## Licença

Este projeto foi desenvolvido para fins acadêmicos na UFC Campus Sobral. Consulte os autores para uso fora do contexto educacional.
