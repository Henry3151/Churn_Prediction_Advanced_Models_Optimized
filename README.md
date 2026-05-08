# 🔄 Churn Prediction — Telecom Customer Churn

> Projeto de previsão de churn de clientes de telecomunicações com **Rede Neural MLP (PyTorch)**, pipeline **MLOps completo**, arquitetura limpa e rastreamento avançado via **MLflow**.  
> Desenvolvido para o **Tech Challenge FIAP + Alura**.

---

## 📊 Resultados do Modelo

| Métrica          | Valor    |
|------------------|----------|
| AUC-ROC          | **0.93** |
| PR-AUC           | **0.88** |
| F1-Score         | **0.85** |
| Recall (Churn)   | **0.87** |
| Precisão (Churn) | **0.83** |

> Modelo final: **MLP PyTorch** com early stopping, treinado com cross-validation estratificada (5-fold) e otimização de hiperparâmetros via Optuna.

---

## 🏗️ Arquitetura do Projeto (Clean Architecture)

```
churn-prediction/
│
├── data/
│   ├── raw/                    # Dados brutos (não versionados)
│   └── processed/              # Dados tratados (não versionados)
│
├── notebooks/
│   ├── 01_eda_baselines.ipynb  # Análise exploratória + baselines
│   └── 02_model_evaluation.ipynb # Avaliação do modelo final
│
├── src/
│   ├── data/
│   │   └── make_dataset.py     # Ingestão e limpeza dos dados
│   ├── features/
│   │   └── build_features.py   # Engenharia de features + preprocessor
│   └── models/
│       └── train_model.py      # Treinamento MLP PyTorch + MLflow
│
├── models/                     # Artefatos treinados (não versionados)
│   └── preprocessor.joblib
│
├── tests/
│   ├── test_smoke.py           # Smoke tests
│   ├── test_schema.py          # Validação de schema
│   └── test_api.py             # Testes de API
│
├── mlflow.db                   # Backend SQLite MLflow (não versionado)
├── mlruns/                     # Experimentos MLflow (não versionados)
├── requirements.txt
├── Makefile
└── README.md
```

---

## 🔁 Pipeline Completo

```
Dados Brutos (Telco CSV)
        │
        ▼
┌─────────────────────┐
│   make_dataset.py   │  ← Limpeza, nulos, tipos, target encoding
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  build_features.py  │  ← StandardScaler, OHE, Pipeline sklearn
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│   train_model.py    │  ← MLP PyTorch, Early Stopping, Optuna
│                     │  ← MLflow logging: params, metrics, artefatos
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  02_model_eval.ipynb│  ← AUC-ROC, PR-AUC, F1, matriz de confusão
└─────────────────────┘
```

---

## 🧪 Análise Exploratória (EDA)

Realizada no notebook `01_eda_baselines.ipynb`:

- **Volume e distribuição**: análise do desbalanceamento de classes (~26% churn)
- **Qualidade dos dados**: tratamento de nulos, outliers e inconsistências
- **Correlações**: features mais preditivas para churn (`tenure`, `Contract`, `TotalCharges`)
- **Baselines**: Dummy Classifier e Regressão Logística como referência
- **Métricas**: AUC-ROC, PR-AUC e F1 como métricas principais

---

## 🤖 Modelagem

### MLP PyTorch
- Arquitetura: `Input → BatchNorm → Dense(128) → ReLU → Dropout(0.3) → Dense(64) → ReLU → Dense(1) → Sigmoid`
- Loss: `BCEWithLogitsLoss` com `pos_weight` para desbalanceamento
- Otimizador: `Adam` com `lr=1e-3`
- Early Stopping: paciência de 10 épocas monitorando `val_loss`
- Cross-validation: estratificada 5-fold

### Comparação de Modelos
| Modelo              | AUC-ROC |     F1   | PR-AUC   |
|---------------------|---------|----------|----------|
| Dummy Classifier    |  0.50   |  0.21    |  0.26    |
| Logistic Regression |  0.84   |  0.72    |  0.78    |
| **MLP PyTorch**     |**0.93** | **0.85** | **0.88** |

---

## 📦 MLflow — Rastreamento de Experimentos

Experimento: `Churn_Prediction_Advanced_Models_Optimized`

Métricas rastreadas:
- `train_loss`, `val_loss` por época
- `auc_roc`, `pr_auc`, `f1`, `recall`, `precision`
- `best_epoch`, `total_params`

Artefatos salvos:
- `preprocessor.joblib`
- `model.pt` (pesos PyTorch)
- `confusion_matrix.png`
- `roc_curve.png`

Para visualizar:
```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```
Acesse: http://127.0.0.1:5000

---

## 🚀 Como Reproduzir

### 1. Clone o repositório
```bash
git clone https://github.com/Henry3151/Churn_Prediction_Advanced_Models_Optimized.git
cd Churn_Prediction_Advanced_Models_Optimized
```

### 2. Crie o ambiente virtual
```bash
python -m venv .venv
.venv\Scripts\activate      # Windows
source .venv/bin/activate   # Linux/Mac
```

### 3. Instale as dependências
```bash
pip install -r requirements.txt
```

### 4. Execute o pipeline
```bash
python src/data/make_dataset.py
python src/features/build_features.py
python src/models/train_model.py
```

### 5. Avalie o modelo
Abra o notebook `notebooks/02_model_evaluation.ipynb` e execute todas as células.

### 6. Visualize os experimentos
```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

---

## 🛠️ Stack Tecnológica

|   Categoria         | Tecnologia          |
|---------------------|---------------------|
| Linguagem           | Python 3.14         |
| Deep Learning       | PyTorch             |
| ML Pipeline         | scikit-learn        |
| Experiment Tracking | MLflow              |
| Otimização          | Optuna              |
| Dados               | pandas, numpy       |
| Visualização        | matplotlib, seaborn |
| Qualidade de código | ruff                |
| Testes              | pytest              |
| Versionamento       | Git + GitHub        |

---

## 📋 Boas Práticas Aplicadas

- ✅ **Reprodutibilidade**: seeds fixos em todo o pipeline (`SEED = 42`)
- ✅ **Logging estruturado**: sem `print()`, usando `logging` em todos os módulos
- ✅ **Clean Architecture**: separação clara entre dados, features e modelos
- ✅ **Cross-validation estratificada**: preserva proporção de classes
- ✅ **Rastreamento completo**: todos os experimentos versionados no MLflow
- ✅ **Sem data leakage**: preprocessor fitado apenas no treino

---

## 👤 Autor

**Henry** — [@Henry3151](https://github.com/Henry3151)  
Tech Challenge · FIAP + Alura · 2026


