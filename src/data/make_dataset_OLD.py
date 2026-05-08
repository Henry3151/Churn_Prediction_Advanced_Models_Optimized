# -*- coding: utf-8 -*-
import logging
import pandas as pd
from pathlib import Path

logger = logging.getLogger(__name__)

def load_data(data_path: Path) -> pd.DataFrame:
    """
    Carrega o dataset de churn de clientes, tentando diferentes codificacoes e delimitadores.

    Args:
        data_path (Path): Caminho para o arquivo CSV do dataset.

    Returns:
        pd.DataFrame: DataFrame carregado.
    """
    encodings = ['utf-8', 'latin1', 'iso-8859-1']
    delimiters = [',', ';', '\t'] # Virgula, ponto e virgula, tab

    for encoding in encodings:
        for delimiter in delimiters:
            try:
                df = pd.read_csv(data_path, encoding=encoding, sep=delimiter)
                # Verificar se o DataFrame tem mais de uma coluna (indicando que o delimitador funcionou)
                if df.shape[1] > 1:
                    logger.info({
                        "event": "data_load_success",
                        "path": str(data_path),
                        "shape": str(df.shape),
                        "encoding": encoding,
                        "delimiter": delimiter,
                        "message": f"Dataset carregado com sucesso (encoding: {encoding}, delimiter: '{delimiter}')."
                    })
                    return df
                else:
                    logger.warning({
                        "event": "data_load_attempt_single_column",
                        "path": str(data_path),
                        "encoding": encoding,
                        "delimiter": delimiter,
                        "message": f"Dataset lido como uma unica coluna com encoding {encoding} e delimiter '{delimiter}'. Tentando proximo."
                    })
            except (UnicodeDecodeError, pd.errors.ParserError) as e:
                logger.debug({
                    "event": "data_load_attempt_failed",
                    "path": str(data_path),
                    "encoding": encoding,
                    "delimiter": delimiter,
                    "error": str(e),
                    "message": f"Tentativa de carregamento falhou com encoding {encoding} e delimiter '{delimiter}'. Erro: {e}"
                })
            except Exception as e:
                logger.error({
                    "event": "data_load_unexpected_error",
                    "path": str(data_path),
                    "error": str(e),
                    "message": "Erro inesperado ao tentar carregar o dataset."
                })
    raise Exception(f"Nao foi possivel carregar o arquivo {data_path} com as codificacoes e delimitadores testados.")

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Realiza a limpeza e pre-processamento inicial do DataFrame.

    Args:
        df (pd.DataFrame): DataFrame bruto.

    Returns:
        pd.DataFrame: DataFrame limpo.
    """
    df_cleaned = df.copy()

    # Remover coluna 'customerID' se existir
    if 'customerID' in df_cleaned.columns:
        df_cleaned = df_cleaned.drop('customerID', axis=1)
        logger.info({"event": "column_removed", "column": "customerID", "message": "Coluna 'customerID' removida."})

    # Converter 'TotalCharges' para numérico, preenchendo nulos com 0
    if 'TotalCharges' in df_cleaned.columns:
        df_cleaned['TotalCharges'] = pd.to_numeric(df_cleaned['TotalCharges'], errors='coerce')
        df_cleaned['TotalCharges'] = df_cleaned['TotalCharges'].fillna(0)
        logger.info({"event": "totalcharges_converted", "message": "Coluna 'TotalCharges' convertida para numerico e nulos preenchidos com 0."})

    # Codificar 'Churn' para binario
    if 'Churn' in df_cleaned.columns:
        df_cleaned['Churn'] = df_cleaned['Churn'].map({'Yes': 1, 'No': 0})
        logger.info({"event": "target_encoded", "message": "Coluna 'Churn' codificada para 0 e 1."})

    # Simplificar categorias em algumas colunas
    for col in ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']:
        if col in df_cleaned.columns:
            df_cleaned[col] = df_cleaned[col].replace({'No internet service': 'No', 'No phone service': 'No'})
            logger.info({"event": "categories_simplified", "column": col, "message": f"Categorias 'No internet service' e 'No phone service' simplificadas para 'No' na coluna {col}."})

    logger.info({"event": "data_cleaned", "shape": str(df_cleaned.shape), "message": "Dados limpos com sucesso."})
    return df_cleaned

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='{"time": "%(asctime)s", "level": "%(levelname)s", "message": "%(message)s"}',
        datefmt='%Y-%m-%dT%H:%M:%SZ'
    )
    logger.info({"event": "make_dataset_script_start", "message": "Iniciando script make_dataset.py"})

    project_dir = Path(__file__).resolve().parents[2]
    raw_data_path = project_dir / "data" / "raw" / "telco_customer_churn.csv"
    processed_data_path = project_dir / "data" / "processed" / "telco_customer_churn_cleaned.csv"

    try:
        df_raw = load_data(raw_data_path)
        df_cleaned = clean_data(df_raw)

        # Salvar o DataFrame limpo
        processed_data_path.parent.mkdir(parents=True, exist_ok=True)
        df_cleaned.to_csv(processed_data_path, index=False)
        logger.info({"event": "processed_data_save_success", "path": str(processed_data_path), "shape": str(df_cleaned.shape), "message": "DataFrame limpo salvo com sucesso."})
    except Exception as e:
        logger.critical({"event": "make_dataset_script_failed", "error": str(e), "message": "Script make_dataset.py falhou."})

    logger.info({"event": "make_dataset_script_end", "message": "Script make_dataset.py concluido."})