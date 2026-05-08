<div align="center">

<img src="https://img.shields.io/badge/Python-3.14-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
<img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white"/>
<img src="https://img.shields.io/badge/MLflow-0194E2?style=for-the-badge&logo=mlflow&logoColor=white"/>
<img src="https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white"/>
<img src="https://img.shields.io/badge/Optuna-4B8BF5?style=for-the-badge&logo=optuna&logoColor=white"/>

<br/><br/>

# 🔄 Churn Prediction — Telecom

### Previsão de churn com Rede Neural MLP (PyTorch) + Pipeline MLOps Completo

*Tech Challenge · FIAP + Alura · 2026*

<br/>

[![AUC-ROC](https://img.shields.io/badge/AUC--ROC-0.93-brightgreen?style=flat-square)](.)
[![PR-AUC](https://img.shields.io/badge/PR--AUC-0.88-brightgreen?style=flat-square)](.)
[![F1-Score](https://img.shields.io/badge/F1--Score-0.85-brightgreen?style=flat-square)](.)
[![Recall](https://img.shields.io/badge/Recall%20(Churn)-0.87-brightgreen?style=flat-square)](.)
[![License](https://img.shields.io/badge/License-MIT-blue?style=flat-square)](.)

</div>

---

## 📋 Sumário

- [Resultados](#-resultados-do-modelo)
- [Arquitetura Clean](#-arquitetura-clean-architecture)
- [Pipeline de Dados](#-pipeline-de-tratamento-de-dados)
- [Pipeline Completo](#-pipeline-completo)
- [EDA](#-análise-exploratória-eda)
- [Modelagem](#-modelagem)
- [MLflow](#-mlflow--rastreamento-de-experimentos)
- [Como Reproduzir](#-como-reproduzir)
- [Stack](#%EF%B8%8F-stack-tecnológica)
- [Boas Práticas](#-boas-práticas-aplicadas)

---

## 📊 Resultados do Modelo

<div align="center">

| Métrica | Baseline (LogReg) | **MLP PyTorch** |
|:---:|:---:|:---:|
| AUC-ROC | 0.84 | **✅ 0.93** |
| PR-AUC | 0.78 | **✅ 0.88** |
| F1-Score | 0.72 | **✅ 0.85** |
| Recall (Churn) | — | **✅ 0.87** |
| Precisão (Churn) | — | **✅ 0.83** |

</div>

> **Modelo final:** MLP PyTorch com early stopping, treinado com cross-validation estratificada (5-fold) e otimização de hiperparâmetros via Optuna.

---

## 🏗️ Arquitetura Clean Architecture

```mermaid
graph TD
    subgraph PROJETO["📁 churn-prediction/"]
        direction TB

        subgraph DATA["📂 data/"]
            RAW["📁 raw/\nDados brutos"]
            PROC["📁 processed/\nDados tratados"]
        end

        subgraph NOTEBOOKS["📓 notebooks/"]
            NB1["01_eda_baselines.ipynb\nEDA + Baselines"]
            NB2["02_model_evaluation.ipynb\nAvaliação Final"]
        end

        subgraph SRC["📂 src/  ← Core da Aplicação"]
            direction LR
            subgraph D["data/"]
                MK["make_dataset.py\nIngestão & Limpeza"]
            end
            subgraph F["features/"]
                BF["build_features.py\nEngenharia de Features"]
            end
            subgraph M["models/"]
                TR["train_model.py\nMLP PyTorch + MLflow"]
            end
            D --> F --> M
        end

        subgraph TESTS["🧪 tests/"]
            T1["test_smoke.py"]
            T2["test_schema.py"]
            T3["test_api.py"]
        end

        subgraph ARTIFACTS["📦 models/"]
            PP["preprocessor.joblib"]
        end

        subgraph MLFLOW_DIR["📈 mlruns/ + mlflow.db"]
            ML["Experimentos MLflow\nSQLite Backend"]
        end

        CONFIG["📄 requirements.txt\nMakefile · README.md"]
    end

    style PROJETO fill:#f8faff,stroke:#4a6fa5,color:#111111
    style SRC fill:#dbeafe,stroke:#2563eb,color:#111111
    style DATA fill:#dcfce7,stroke:#16a34a,color:#111111
    style NOTEBOOKS fill:#ede9fe,stroke:#7c3aed,color:#111111
    style TESTS fill:#fee2e2,stroke:#dc2626,color:#111111
    style ARTIFACTS fill:#fef9c3,stroke:#ca8a04,color:#111111
    style MLFLOW_DIR fill:#cffafe,stroke:#0891b2,color:#111111
```

---

## 🔬 Pipeline de Tratamento de Dados

```mermaid
flowchart TD
    A([📥 Telco CSV\nDados Brutos]) --> B

    subgraph B["🧹 make_dataset.py — Ingestão & Limpeza"]
        B1[Carrega CSV com pandas] --> B2[Converte TotalCharges\npara numérico]
        B2 --> B3[Remove / imputa\nvalores nulos]
        B3 --> B4[Padroniza tipos\nde colunas]
        B4 --> B5[Encode do target\nChurn → 0 / 1]
    end

    B --> C

    subgraph C["⚙️ build_features.py — Engenharia de Features"]
        C1[Separa features\nnuméricas e categóricas] --> C2[StandardScaler\nem colunas numéricas]
        C2 --> C3[OneHotEncoder\nem colunas categóricas]
        C3 --> C4[ColumnTransformer\nsklearn Pipeline]
        C4 --> C5[Fit APENAS no treino\n⚠️ Sem data leakage]
        C5 --> C6[Salva preprocessor\n.joblib]
    end

    C --> D

    subgraph D["✂️ Divisão Estratificada"]
        D1[Train / Validation / Test\npreserva ~26% churn] --> D2[5-Fold Cross-Validation\nno conjunto de treino]
    end

    D --> E([✅ Features Prontas\npara Modelagem])

    style A fill:#dcfce7,stroke:#16a34a,color:#111111
    style B fill:#dbeafe,stroke:#2563eb,color:#111111
    style C fill:#ede9fe,stroke:#7c3aed,color:#111111
    style D fill:#fef9c3,stroke:#ca8a04,color:#111111
    style E fill:#dcfce7,stroke:#16a34a,color:#111111
```

---

## 🔁 Pipeline Completo

```mermaid
flowchart LR
    A([🗂️ Dados Brutos\nTelco CSV])
    --> B[🧹 make_dataset.py\nLimpeza & Target Encoding]
    --> C[⚙️ build_features.py\nScaler + OHE + Pipeline]
    --> D[🤖 train_model.py\nMLP PyTorch + Optuna]
    --> E[📈 MLflow Tracking\nParams · Metrics · Artefatos]
    --> F[📓 model_evaluation.ipynb\nAUC-ROC · PR-AUC · F1]

    style A fill:#dcfce7,stroke:#16a34a,color:#111111
    style B fill:#dbeafe,stroke:#2563eb,color:#111111
    style C fill:#ede9fe,stroke:#7c3aed,color:#111111
    style D fill:#fee2e2,stroke:#dc2626,color:#111111
    style E fill:#cffafe,stroke:#0891b2,color:#111111
    style F fill:#fef9c3,stroke:#ca8a04,color:#111111
```

---

## 🔍 Análise Exploratória (EDA)

> Realizada no notebook `01_eda_baselines.ipynb`

| Etapa | Descrição |
|---|---|
| 📊 **Volume e distribuição** | Análise do desbalanceamento de classes (~26% churn) |
| 🔎 **Qualidade dos dados** | Tratamento de nulos, outliers e inconsistências |
| 🔗 **Correlações** | Features mais preditivas: `tenure`, `Contract`, `TotalCharges` |
| 📏 **Baselines** | Dummy Classifier e Regressão Logística como referência |
| 📐 **Métricas** | AUC-ROC, PR-AUC e F1 como métricas principais |

---

## 🤖 Modelagem

### Arquitetura MLP PyTorch

```mermaid
graph LR
    IN([Input\nFeatures]) --> BN[BatchNorm]
    BN --> D1[Dense 128\n+ ReLU]
    D1 --> DR[Dropout 0.3]
    DR --> D2[Dense 64\n+ ReLU]
    D2 --> D3[Dense 1]
    D3 --> SIG([Sigmoid\nOutputChurn Prob])

    style IN fill:#dcfce7,stroke:#16a34a,color:#111111
    style SIG fill:#dcfce7,stroke:#16a34a,color:#111111
    style BN fill:#dbeafe,stroke:#2563eb,color:#111111
    style D1 fill:#ede9fe,stroke:#7c3aed,color:#111111
    style DR fill:#fee2e2,stroke:#dc2626,color:#111111
    style D2 fill:#ede9fe,stroke:#7c3aed,color:#111111
    style D3 fill:#ede9fe,stroke:#7c3aed,color:#111111
```

| Componente | Detalhe |
|---|---|
| **Loss** | `BCEWithLogitsLoss` com `pos_weight` para desbalanceamento |
| **Otimizador** | `Adam` com `lr=1e-3` |
| **Early Stopping** | Paciência de 10 épocas monitorando `val_loss` |
| **Cross-validation** | Estratificada 5-fold |
| **Hiperparâmetros** | Otimizados via **Optuna** |

### Comparação de Modelos

| Modelo | AUC-ROC | F1 | PR-AUC |
|:---:|:---:|:---:|:---:|
| Dummy Classifier | 0.50 | 0.21 | 0.26 |
| Logistic Regression | 0.84 | 0.72 | 0.78 |
| **MLP PyTorch** ⭐ | **0.93** | **0.85** | **0.88** |

---

## 📦 MLflow — Rastreamento de Experimentos

> Experimento: `Churn_Prediction_Advanced_Models_Optimized`

<details>
<summary><b>Métricas e Artefatos Rastreados</b></summary>

**Métricas por época:**
- `train_loss`, `val_loss`
- `auc_roc`, `pr_auc`, `f1`, `recall`, `precision`
- `best_epoch`, `total_params`

**Artefatos salvos:**
- `preprocessor.joblib`
- `model.pt` (pesos PyTorch)
- `confusion_matrix.png`
- `roc_curve.png`

</details>

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
# Acesse: http://127.0.0.1:5000
```

---

## 🚀 Como Reproduzir

```bash
# 1. Clone o repositório
git clone https://github.com/Henry3151/Churn_Prediction_Advanced_Models_Optimized.git
cd Churn_Prediction_Advanced_Models_Optimized

# 2. Ambiente virtual
python -m venv .venv
source .venv/bin/activate        # Linux/Mac
# .venv\Scripts\activate         # Windows

# 3. Dependências
pip install -r requirements.txt

# 4. Execute o pipeline
python src/data/make_dataset.py
python src/features/build_features.py
python src/models/train_model.py

# 5. Visualize os experimentos
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

> Avalie o modelo abrindo `notebooks/02_model_evaluation.ipynb` e executando todas as células.

---

## 🛠️ Stack Tecnológica

<div align="center">

| Categoria | Tecnologia |
|:---:|:---:|
| Linguagem | ![Python](https://img.shields.io/badge/Python_3.14-3776AB?style=flat-square&logo=python&logoColor=white) |
| Deep Learning | ![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white) |
| ML Pipeline | ![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikitlearn&logoColor=white) |
| Experiment Tracking | ![MLflow](https://img.shields.io/badge/MLflow-0194E2?style=flat-square&logo=mlflow&logoColor=white) |
| Otimização | ![Optuna](https://img.shields.io/badge/Optuna-4B8BF5?style=flat-square&logoColor=white) |
| Dados | ![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat-square&logo=pandas&logoColor=white) ![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white) |
| Visualização | ![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=flat-square&logoColor=white) ![Seaborn](https://img.shields.io/badge/Seaborn-4C72B0?style=flat-square&logoColor=white) |
| Qualidade de código | ![Ruff](https://img.shields.io/badge/Ruff-D7FF64?style=flat-square&logoColor=black) |
| Testes | ![pytest](https://img.shields.io/badge/pytest-0A9EDC?style=flat-square&logo=pytest&logoColor=white) |
| Versionamento | ![Git](https://img.shields.io/badge/Git-F05032?style=flat-square&logo=git&logoColor=white) ![GitHub](https://img.shields.io/badge/GitHub-181717?style=flat-square&logo=github&logoColor=white) |

</div>

---

## ✅ Boas Práticas Aplicadas

```
✅  Reprodutibilidade    → Seeds fixos em todo o pipeline (SEED = 42)
✅  Logging estruturado  → Sem print(), usando logging em todos os módulos
✅  Clean Architecture   → Separação clara entre dados, features e modelos
✅  Cross-validation     → Estratificada, preserva proporção de classes
✅  Rastreamento MLflow  → Todos os experimentos versionados
✅  Sem data leakage     → Preprocessor fitado apenas no conjunto de treino
```

---

<div align="center">

**Desenvolvido por [Henrique Silva](https://github.com/Henry3151)**  
*Tech Challenge · FIAP + Alura · 2026*

</div>
