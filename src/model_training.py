"""
model_training.py

Módulo para definición y entrenamiento de modelos supervisados.
Incluye: Logistic Regression, Decision Trees, Random Forest, SVM, KNN, Gradient Boosting.
Con soporte completo para entrenamiento, predicción y persistencia.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib


def create_logistic_regression(random_state: int = 42, max_iter: int = 1000) -> Pipeline:
    """
    Crea pipeline con StandardScaler + Logistic Regression.
    
    Args:
        random_state (int): Seed para reproducibilidad
        max_iter (int): Número máximo de iteraciones
        
    Returns:
        Pipeline: Pipeline con modelo
    """
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', LogisticRegression(random_state=random_state, max_iter=max_iter))
    ])
    return pipeline


def create_decision_tree(random_state: int = 42, max_depth: int = None) -> Pipeline:
    """
    Crea pipeline con Decision Tree.
    
    Args:
        random_state (int): Seed para reproducibilidad
        max_depth (int): Profundidad máxima del árbol
        
    Returns:
        Pipeline: Pipeline con modelo
    """
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', DecisionTreeClassifier(random_state=random_state, max_depth=max_depth))
    ])
    return pipeline


def create_random_forest(random_state: int = 42, n_estimators: int = 100, max_depth: int = None) -> Pipeline:
    """
    Crea pipeline con Random Forest.
    
    Args:
        random_state (int): Seed para reproducibilidad
        n_estimators (int): Número de árboles
        max_depth (int): Profundidad máxima
        
    Returns:
        Pipeline: Pipeline con modelo
    """
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', RandomForestClassifier(random_state=random_state, n_estimators=n_estimators, max_depth=max_depth))
    ])
    return pipeline


def create_svm(kernel: str = 'rbf', random_state: int = 42, C: float = 1.0) -> Pipeline:
    """
    Crea pipeline con SVM.
    
    Args:
        kernel (str): Tipo de kernel ('rbf', 'linear', 'poly')
        random_state (int): Seed para reproducibilidad
        C (float): Parámetro de regularización
        
    Returns:
        Pipeline: Pipeline con modelo
    """
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', SVC(kernel=kernel, random_state=random_state, probability=True, C=C))
    ])
    return pipeline


def create_knn(n_neighbors: int = 5) -> Pipeline:
    """
    Crea pipeline con KNN.
    
    Args:
        n_neighbors (int): Número de vecinos
        
    Returns:
        Pipeline: Pipeline con modelo
    """
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', KNeighborsClassifier(n_neighbors=n_neighbors))
    ])
    return pipeline


def create_gradient_boosting(random_state: int = 42, n_estimators: int = 100, 
                            learning_rate: float = 0.1, max_depth: int = 3) -> Pipeline:
    """
    Crea pipeline con Gradient Boosting.
    
    Args:
        random_state (int): Seed para reproducibilidad
        n_estimators (int): Número de estimadores
        learning_rate (float): Tasa de aprendizaje
        max_depth (int): Profundidad máxima
        
    Returns:
        Pipeline: Pipeline con modelo
    """
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', GradientBoostingClassifier(random_state=random_state, n_estimators=n_estimators,
                                            learning_rate=learning_rate, max_depth=max_depth))
    ])
    return pipeline


def train_model(model: Pipeline, X_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
    """
    Entrena un modelo.
    
    Args:
        model (Pipeline): Pipeline del modelo
        X_train (pd.DataFrame): Features de entrenamiento
        y_train (pd.Series): Target de entrenamiento
        
    Returns:
        Pipeline: Modelo entrenado
    """
    model.fit(X_train, y_train)
    return model


def predict(model: Pipeline, X: pd.DataFrame) -> np.ndarray:
    """
    Realiza predicciones con un modelo entrenado.
    
    Args:
        model (Pipeline): Modelo entrenado
        X (pd.DataFrame): Features
        
    Returns:
        np.ndarray: Predicciones
    """
    return model.predict(X)


def predict_proba(model: Pipeline, X: pd.DataFrame) -> np.ndarray:
    """
    Obtiene probabilidades de predicción.
    
    Args:
        model (Pipeline): Modelo entrenado
        X (pd.DataFrame): Features
        
    Returns:
        np.ndarray: Probabilidades
    """
    if hasattr(model.named_steps['model'], 'predict_proba'):
        return model.predict_proba(X)
    else:
        raise ValueError("El modelo no soporta predict_proba")


def save_model(model: Pipeline, filepath: str) -> None:
    """
    Guarda un modelo entrenado.
    
    Args:
        model (Pipeline): Modelo a guardar
        filepath (str): Ruta donde guardar el modelo
    """
    joblib.dump(model, filepath)
    print(f"✅ Modelo guardado en: {filepath}")


def load_model(filepath: str) -> Pipeline:
    """
    Carga un modelo previamente guardado.
    
    Args:
        filepath (str): Ruta del modelo guardado
        
    Returns:
        Pipeline: Modelo cargado
    """
    return joblib.load(filepath)


def get_all_models(random_state: int = 42) -> dict:
    """
    Retorna diccionario con todos los modelos disponibles.
    
    Args:
        random_state (int): Seed para reproducibilidad
        
    Returns:
        dict: Diccionario {nombre_modelo: pipeline}
    """
    models = {
        'Logistic Regression': create_logistic_regression(random_state=random_state),
        'Decision Tree': create_decision_tree(random_state=random_state),
        'Random Forest': create_random_forest(random_state=random_state),
        'SVM': create_svm(random_state=random_state),
        'KNN': create_knn(n_neighbors=5),
        'Gradient Boosting': create_gradient_boosting(random_state=random_state)
    }
    return models
