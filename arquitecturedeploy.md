# Arquitetura de Deploy — Churn Prediction

## Visão Geral

Modelo de predição de churn para clientes de uma operadora de telecomunicações, servido via API REST em tempo real. O objetivo é acionar campanhas de retenção **antes** do cancelamento, expondo o risco do cliente durante o atendimento no CRM.

---

## Decisão: Real-Time vs Batch

### Opção escolhida — Inferência em Tempo Real

A API FastAPI expõe o modelo via HTTP. Cada requisição recebe as features de um cliente (ou lote de até 500) e retorna a probabilidade de churn imediatamente.

```
Cliente / Sistema CRM
        |
        v
  [ API FastAPI ]          uvicorn src.api.main:app
        |
        v
  [ Preprocessor ]         models/preprocessor.joblib
        |
        v
  [ MLP PyTorch ]          models/mlp_model.pt
        |
        v
  Resposta JSON
  { churn_probability, churn_prediction, risk_level }
```

### Justificativa

| Critério | Avaliação |
|---|---|
| **Latência exigida** | Baixa — o CRM precisa exibir o risco durante o atendimento |
| **Volume de requisições** | Moderado — não justifica infraestrutura de streaming |
| **Frequência de predição** | Sob demanda — acionada quando o agente abre o perfil |
| **Custo de infraestrutura** | Baixo — um único processo uvicorn suporta o volume esperado |
| **Complexidade operacional** | Baixa — sem necessidade de orquestrador de jobs |

### Fluxo de produção

```
1. Agente de atendimento abre perfil do cliente no CRM
2. CRM chama POST /predict com as features do cliente
3. API retorna { churn_probability, risk_level } em < 100ms
4. CRM exibe badge de risco e sugere script de retenção
```

---

## Endpoints

| Rota | Método | Uso |
|---|---|---|
| `/health` | GET | Health check — monitoramento e liveness probe |
| `/predict` | POST | Predição individual — integração com CRM em tempo real |
| `/predict/batch` | POST | Predição em lote — até 500 clientes por requisição |

### Exemplo de request — `/predict`

```json
POST /predict
{
  "customer": {
    "gender": "Male",
    "SeniorCitizen": 0,
    "tenure": 12,
    "MonthlyCharges": 70.35,
    "TotalCharges": 845.50,
    "Contract": "Month-to-month",
    "InternetService": "Fiber optic",
    "PaymentMethod": "Electronic check",
    ...
  },
  "threshold": 0.5
}
```

### Exemplo de response

```json
{
  "churn_probability": 0.7823,
  "churn_prediction": true,
  "threshold_used": 0.5,
  "model_version": "mlp_model",
  "risk_level": "high"
}
```

### Níveis de risco

| `risk_level` | Faixa de probabilidade |
|---|---|
| `low` | < 0.30 |
| `medium` | 0.30 – 0.59 |
| `high` | ≥ 0.60 |

---

## Arquitetura do Modelo

### MLP PyTorch

```
Input (19 features processadas)
        |
  Linear(19 → 128) → BatchNorm → ReLU → Dropout(0.3)
        |
  Linear(128 → 64) → BatchNorm → ReLU → Dropout(0.3)
        |
  Linear(64 → 32)  → BatchNorm → ReLU → Dropout(0.3)
        |
  Linear(32 → 1)   → Sigmoid
        |
  churn_probability ∈ [0, 1]
```

### Treinamento

| Parâmetro | Valor |
|---|---|
| `hidden_dims` | [128, 64, 32] |
| `dropout_rate` | 0.3 |
| `learning_rate` | 1e-3 |
| `batch_size` | 64 |
| `max_epochs` | 150 |
| `patience` (early stopping) | 15 |
| `optimizer` | Adam + weight_decay=1e-5 |
| `loss` | BCEWithLogitsLoss (pos_weight para desbalanceamento) |
| `scheduler` | ReduceLROnPlateau (patience=5, factor=0.5) |

### Threshold ótimo

O threshold é definido por análise de custo FP/FN:

| Custo | Valor padrão |
|---|---|
| Falso Negativo (cliente perdido) | R$ 500 |
| Falso Positivo (retenção desnecessária) | R$ 50 |

O threshold ótimo é calculado no treino e registrado no MLflow. O valor padrão na API é `0.5`, sobrescritível por requisição.

---

## Estrutura do Projeto

```
churn-prediction/
├── .env
├── pyproject.toml
├── requirements.txt
├── Dockerfile
├── mlflow.db
│
├── models/
│   ├── mlp_model.pt          # pesos da MLP treinada
│   └── preprocessor.joblib   # pipeline de pré-processamento
│
├── src/
│   ├── api/
│   │   ├── __init__.py
│   │   └── main.py           # FastAPI — endpoints /health /predict /predict/batch
│   ├── data/
│   │   └── make_dataset.py
│   ├── features/
│   │   └── build_features.py
│   └── models/
│       └── train_model.py
│
├── notebooks/
│   ├── 01_eda_baselines.ipynb
│   └── 02_model_evaluation.ipynb
│
└── tests/
    └── test_churn.py         # 22 testes — modelo, schema, API
```

---

## Arquitetura de Produção Recomendada

```
                    +------------------+
                    |   Load Balancer  |
                    +--------+---------+
                             |
          +------------------+------------------+
          |                                     |
+---------+--------+                 +----------+--------+
|  API Instance 1  |                 |  API Instance 2   |
|  FastAPI/Uvicorn |                 |  FastAPI/Uvicorn  |
+------------------+                 +-------------------+
          |                                     |
          +------------------+------------------+
                             |
                +------------+-----------+
                |  Shared Model Storage  |
                |  preprocessor.joblib   |
                |  mlp_model.pt          |
                +------------------------+
```

### Configuração mínima

| Componente | Recomendação |
|---|---|
| **Servidor** | Uvicorn com `workers = 2 * CPU cores + 1` |
| **Reverse proxy** | Nginx ou Traefik |
| **Health check** | `GET /health` a cada 30s |
| **Logs** | JSON estruturado — ingestão no ELK ou CloudWatch |
| **Modelo** | Carregado em memória no startup — não relido a cada requisição |
| **Timeout** | 5s por requisição (P99 esperado < 100ms) |

### Deploy com Docker

```bash
# Build
docker build -t churn-api .

# Run
docker run -p 8000:8000 churn-api
```

---

## Processamento Batch (uso complementar)

O modo batch não é o fluxo principal, mas complementa o real-time para:

- **Relatórios gerenciais** — ranking diário de clientes em risco para equipe comercial
- **Campanhas de e-mail** — segmento para campanha de retenção offline
- **Monitoramento de drift** — distribuição das probabilidades comparada ao baseline

---

## Versionamento de Modelo

Toda atualização do modelo deve seguir o processo:

1. Registrar novo run no MLflow com todas as métricas
2. Salvar `mlp_model.pt` e `preprocessor.joblib` juntos (são dependentes)
3. Atualizar `model_version` no `AppState` para rastreabilidade nas respostas da API
4. Executar suite de testes (`pytest tests/ -v`) antes do deploy
5. Fazer deploy com zero-downtime (blue-green ou rolling update)

### Métricas registradas no MLflow

| Métrica | Descrição |
|---|---|
| `roc_auc` | AUC-ROC no conjunto de teste |
| `f1_score` | F1-score com threshold=0.5 |
| `pr_auc` | AUC da curva Precision-Recall |
| `best_threshold` | Threshold ótimo por análise de custo |
| `best_total_cost` | Custo total mínimo (FP + FN) |
| `train_loss` / `val_loss` | Loss por época |

---

## Status do Projeto

| Componente | Status |
|---|---|
| `preprocessor.joblib` | ✅ Gerado e salvo |
| `mlp_model.pt` | ✅ Gerado e salvo |
| API FastAPI (`/health`, `/predict`, `/predict/batch`) | ✅ Operacional |
| Suite de testes | ✅ 22/22 passando |
| MLflow tracking | ✅ Configurado |
| Dockerfile | ✅ Pronto |