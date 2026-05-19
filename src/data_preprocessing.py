"""
data_preprocessing.py

Módulo para funciones de preparación y limpieza de datos.
Incluye: manejo de valores nulos, duplicados, outliers, encoding, normalización y pipeline completo.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import joblib


def load_data(filepath: str) -> pd.DataFrame:
    """
    Carga datos desde archivo CSV.
    
    Args:
        filepath (str): Ruta del archivo CSV
        
    Returns:
        pd.DataFrame: Dataframe cargado
    """
    return pd.read_csv(filepath)


def handle_missing_values(df: pd.DataFrame, strategy: str = 'drop') -> pd.DataFrame:
    """
    Maneja valores nulos según la estrategia especificada.
    
    Args:
        df (pd.DataFrame): Dataframe con posibles valores nulos
        strategy (str): 'drop' | 'mean' | 'median' | 'ffill' | 'bfill'
        
    Returns:
        pd.DataFrame: Dataframe sin valores nulos
    """
    df_copy = df.copy()
    
    if strategy == 'drop':
        return df_copy.dropna()
    elif strategy == 'mean':
        numeric_cols = df_copy.select_dtypes(include=[np.number]).columns
        df_copy[numeric_cols] = df_copy[numeric_cols].fillna(df_copy[numeric_cols].mean())
        return df_copy
    elif strategy == 'median':
        numeric_cols = df_copy.select_dtypes(include=[np.number]).columns
        df_copy[numeric_cols] = df_copy[numeric_cols].fillna(df_copy[numeric_cols].median())
        return df_copy
    elif strategy == 'ffill':
        return df_copy.fillna(method='ffill')
    elif strategy == 'bfill':
        return df_copy.fillna(method='bfill')
    else:
        return df_copy


def remove_duplicates(df: pd.DataFrame, subset: list = None) -> pd.DataFrame:
    """
    Elimina filas duplicadas.
    
    Args:
        df (pd.DataFrame): Dataframe de entrada
        subset (list): Columnas a considerar para duplicados (None = todas)
        
    Returns:
        pd.DataFrame: Dataframe sin duplicados
    """
    return df.drop_duplicates(subset=subset, keep='first').reset_index(drop=True)


def detect_outliers(df: pd.DataFrame, columns: list, method: str = 'iqr', threshold: float = 1.5) -> pd.DataFrame:
    """
    Detecta y remueve outliers usando IQR o Z-score.
    
    Args:
        df (pd.DataFrame): Dataframe de entrada
        columns (list): Columnas numéricas a analizar
        method (str): 'iqr' o 'zscore'
        threshold (float): Multiplicador para IQR (default 1.5) o Z-score (default 3)
        
    Returns:
        pd.DataFrame: Dataframe con outliers removidos
    """
    df_copy = df.copy()
    
    if method == 'iqr':
        numeric_cols = [col for col in columns if col in df_copy.columns and df_copy[col].dtype in [np.number]]
        if numeric_cols:
            Q1 = df_copy[numeric_cols].quantile(0.25)
            Q3 = df_copy[numeric_cols].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            mask = ~((df_copy[numeric_cols] < lower_bound) | (df_copy[numeric_cols] > upper_bound)).any(axis=1)
            return df_copy[mask].reset_index(drop=True)
    elif method == 'zscore':
        numeric_cols = [col for col in columns if col in df_copy.columns and df_copy[col].dtype in [np.number]]
        if numeric_cols:
            z_scores = np.abs((df_copy[numeric_cols] - df_copy[numeric_cols].mean()) / df_copy[numeric_cols].std())
            mask = ~(z_scores > threshold).any(axis=1)
            return df_copy[mask].reset_index(drop=True)
    
    return df_copy


def encode_categorical(df: pd.DataFrame, categorical_cols: list, method: str = 'label') -> tuple:
    """
    Codifica variables categóricas.
    
    Args:
        df (pd.DataFrame): Dataframe de entrada
        categorical_cols (list): Columnas categóricas a codificar
        method (str): 'label' o 'onehot'
        
    Returns:
        tuple: (Dataframe con variables codificadas, diccionario de encoders)
    """
    df_copy = df.copy()
    
    if method == 'label':
        encoders = {}
        for col in categorical_cols:
            if col in df_copy.columns:
                le = LabelEncoder()
                df_copy[col] = le.fit_transform(df_copy[col].astype(str))
                encoders[col] = le
        return df_copy, encoders
    elif method == 'onehot':
        return pd.get_dummies(df_copy, columns=categorical_cols, drop_first=True), {}
    
    return df_copy, {}


def scale_features(df: pd.DataFrame, numeric_cols: list, method: str = 'standard', fit_scaler=None) -> tuple:
    """
    Normaliza/escala variables numéricas.
    
    Args:
        df (pd.DataFrame): Dataframe de entrada
        numeric_cols (list): Columnas a escalar
        method (str): 'standard' o 'minmax'
        fit_scaler: Scaler ya entrenado (para aplicar a test)
        
    Returns:
        tuple: (Dataframe escalado, objeto scaler)
    """
    df_copy = df.copy()
    numeric_cols = [col for col in numeric_cols if col in df_copy.columns]
    
    if method == 'standard':
        scaler = StandardScaler() if fit_scaler is None else fit_scaler
    else:
        scaler = MinMaxScaler() if fit_scaler is None else fit_scaler
    
    if fit_scaler is None:
        df_copy[numeric_cols] = scaler.fit_transform(df_copy[numeric_cols])
    else:
        df_copy[numeric_cols] = scaler.transform(df_copy[numeric_cols])
    
    return df_copy, scaler


def train_test_split_data(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, 
                          random_state: int = 42, stratify: bool = True) -> tuple:
    """
    Divide datos en train y test.
    
    Args:
        X (pd.DataFrame): Features
        y (pd.Series): Target
        test_size (float): Proporción de datos de test (default 0.2)
        random_state (int): Seed para reproducibilidad
        stratify (bool): Si es True, mantiene proporciones de clases
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    stratify_arg = y if stratify else None
    return train_test_split(X, y, test_size=test_size, random_state=random_state, 
                          stratify=stratify_arg)


def prepare_data_pipeline(df, target_col: str, categorical_cols: list = None, 
                         numeric_cols: list = None, test_size: float = 0.2, 
                         random_state: int = 42) -> tuple:
    """
    Pipeline completo de preparación de datos.
    
    Args:
        df (str o pd.DataFrame): Path a CSV o Dataframe
        target_col (str): Nombre de la columna target
        categorical_cols (list): Columnas categóricas (si None, detecta automáticamente)
        numeric_cols (list): Columnas numéricas (si None, detecta automáticamente)
        test_size (float): Proporción de test
        random_state (int): Seed
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test, scaler, encoders)
    """
    # Cargar datos si es string
    if isinstance(df, str):
        df = load_data(df)
    
    df_clean = df.copy()
    
    # Detectar columnas si no se especifican
    if categorical_cols is None:
        categorical_cols = df_clean.select_dtypes(include=['object']).columns.tolist()
        categorical_cols = [col for col in categorical_cols if col != target_col]
    
    if numeric_cols is None:
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
    
    # Separar target
    y = df_clean[target_col]
    X = df_clean.drop(columns=[target_col])
    
    # Encoding de variables categóricas
    if categorical_cols:
        X_encoded, encoders = encode_categorical(X, categorical_cols, method='label')
    else:
        X_encoded = X
        encoders = {}
    
    # Escalado de variables numéricas
    if numeric_cols:
        X_scaled, scaler = scale_features(X_encoded, numeric_cols, method='standard')
    else:
        X_scaled = X_encoded
        scaler = None
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split_data(X_scaled, y, 
                                                               test_size=test_size, 
                                                               random_state=random_state)
    
    return X_train, X_test, y_train, y_test, scaler, encoders


def save_preprocessing_artifacts(scaler, encoders, filepath: str = 'models/preprocessing'):
    """
    Guarda el scaler y encoders para usar en datos nuevos.
    
    Args:
        scaler: Objeto StandardScaler o MinMaxScaler
        encoders (dict): Diccionario de LabelEncoders
        filepath (str): Ruta base para guardar
    """
    if scaler is not None:
        joblib.dump(scaler, f'{filepath}/scaler.joblib')
    if encoders:
        joblib.dump(encoders, f'{filepath}/encoders.joblib')


def load_preprocessing_artifacts(filepath: str = 'models/preprocessing'):
    """
    Carga el scaler y encoders guardados.
    
    Args:
        filepath (str): Ruta base donde están guardados
        
    Returns:
        tuple: (scaler, encoders)
    """
    scaler = joblib.load(f'{filepath}/scaler.joblib')
    encoders = joblib.load(f'{filepath}/encoders.joblib')
    return scaler, encoders
