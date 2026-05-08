# -*- coding: utf-8 -*-
import logging
import pandas as pd
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)

def split_data(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """
    Divide o DataFrame em conjuntos de treino e teste.

    Args:
        df (pd.DataFrame): DataFrame de entrada.
        test_size (float): Proporcao do dataset para o conjunto de teste.
        random_state (int): Semente para reprodutibilidade.

    Returns:
        tuple: X_train, X_test, y_train, y_test.
    """
    X = df.drop('Churn', axis=1)
    y = df['Churn']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    logger.info({"event": "data_split_success", "X_train_shape": str(X_train.shape), "X_test_shape": str(X_test.shape), "message": "Dados divididos em treino e teste."})
    return X_train, X_test, y_train, y_test

def create_preprocessor(X_train: pd.DataFrame) -> ColumnTransformer:
    """
    Cria um pre-processador para dados numericos e categoricos.

    Args:
        X_train (pd.DataFrame): DataFrame de treino para identificar colunas.

    Returns:
        ColumnTransformer: Pre-processador configurado.
    """
    numeric_features = X_train.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X_train.select_dtypes(include=['object', 'bool']).columns

    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='passthrough' # Manter colunas nao transformadas
    )
    logger.info({"event": "preprocessor_created", "numeric_features": numeric_features.tolist(), "categorical_features": categorical_features.tolist(), "message": "Pre-processador criado."})
    return preprocessor

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='{"time": "%(asctime)s", "level": "%(levelname)s", "message": "%(message)s"}',
        datefmt='%Y-%m-%dT%H:%M:%SZ'
    )
    logger.info({"event": "build_features_script_start", "message": "Iniciando script build_features.py"})

    project_dir = Path(__file__).resolve().parents[2]
    processed_data_path = project_dir / "data" / "processed" / "telco_customer_churn_cleaned.csv"
    preprocessor_path = project_dir / "models" / "preprocessor.joblib"

    try:
        df_cleaned = pd.read_csv(processed_data_path)
        logger.info({"event": "cleaned_data_load_success", "path": str(processed_data_path), "shape": str(df_cleaned.shape), "message": "DataFrame limpo carregado."})

        X_train, X_test, y_train, y_test = split_data(df_cleaned)

        preprocessor = create_preprocessor(X_train)
        preprocessor.fit(X_train) # Fit no conjunto de treino

        # Salvar o preprocessor
        preprocessor_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(preprocessor, preprocessor_path)
        logger.info({"event": "preprocessor_save_success", "path": str(preprocessor_path), "message": "Preprocessor salvo com sucesso."})

        # Opcional: Salvar X_train, X_test, y_train, y_test para uso posterior
        # joblib.dump(X_train, project_dir / "data" / "processed" / "X_train.joblib")
        # joblib.dump(X_test, project_dir / "data" / "processed" / "X_test.joblib")
        # joblib.dump(y_train, project_dir / "data" / "processed" / "y_train.joblib")
        # joblib.dump(y_test, project_dir / "data" / "processed" / "y_test.joblib")
        # logger.info({"event": "split_data_saved", "message": "Conjuntos de treino e teste salvos."})

    except FileNotFoundError:
        logger.critical({"event": "build_features_script_failed", "error": f"Arquivo nao encontrado: {processed_data_path}. Certifique-se de que make_dataset.py foi executado primeiro.", "message": "Script build_features.py falhou."})
        raise
    except Exception as e:
        logger.critical({"event": "build_features_script_failed", "error": str(e), "message": "Script build_features.py falhou."})
        raise

    logger.info({"event": "build_features_script_end", "message": "Script build_features.py concluido."})
