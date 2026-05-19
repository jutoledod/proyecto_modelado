"""
model_evaluation.py

Módulo para evaluación y comparación de modelos.
Incluye: cross-validation, métricas, tablas comparativas, visualizaciones.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import cross_validate, cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    mean_absolute_error, mean_squared_error
)
import matplotlib.pyplot as plt
import seaborn as sns
import joblib


def cross_validation_evaluation(model, X, y, cv: int = 5, scoring: list = None, stratified: bool = True, random_state: int = 42):
    """
    Realiza cross-validation en un modelo con opción de stratificación.
    
    Args:
        model: Modelo a evaluar
        X (pd.DataFrame): Features
        y (pd.Series): Target
        cv (int): Número de folds
        scoring (list): Métricas a calcular
        stratified (bool): Usar StratifiedKFold para mantener proporciones de clases
        random_state (int): Seed para reproducibilidad
        
    Returns:
        dict: Resultados de cross-validation
    """
    if scoring is None:
        scoring = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted']
    
    # Usar StratifiedKFold para garantizar distribución equilibrada en cada fold
    if stratified:
        cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    else:
        cv_splitter = cv
    
    cv_results = cross_validate(model, X, y, cv=cv_splitter, scoring=scoring, return_train_score=True)
    return cv_results


def summarize_cv_results(cv_results: dict) -> dict:
    """
    Resume los resultados de cross-validation.
    
    Args:
        cv_results (dict): Resultados de cross_validate
        
    Returns:
        dict: Resumen con media y desv. est.
    """
    summary = {}
    for metric in cv_results:
        if 'test' in metric:
            metric_name = metric.replace('test_', '')
            scores = cv_results[metric]
            summary[metric_name] = {
                'mean': np.mean(scores),
                'std': np.std(scores),
                'min': np.min(scores),
                'max': np.max(scores)
            }
    return summary


def calculate_metrics_classification(y_true, y_pred, y_pred_proba=None):
    """
    Calcula métricas de clasificación.
    
    Args:
        y_true: Target real
        y_pred: Predicciones
        y_pred_proba: Probabilidades (opcional para AUC)
        
    Returns:
        dict: Diccionario con todas las métricas
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'f1': f1_score(y_true, y_pred, average='weighted', zero_division=0),
    }
    
    if y_pred_proba is not None:
        try:
            # Para clasificación multiclase
            if len(y_pred_proba.shape) > 1 and y_pred_proba.shape[1] > 2:
                metrics['auc_roc'] = roc_auc_score(y_true, y_pred_proba, multi_class='ovr', zero_division=0)
            else:
                metrics['auc_roc'] = roc_auc_score(y_true, y_pred_proba, zero_division=0)
        except Exception as e:
            metrics['auc_roc'] = None
    
    return metrics


def calculate_metrics_regression(y_true, y_pred):
    """
    Calcula métricas de regresión.
    
    Args:
        y_true: Target real
        y_pred: Predicciones
        
    Returns:
        dict: Diccionario con todas las métricas
    """
    mse = mean_squared_error(y_true, y_pred)
    metrics = {
        'mae': mean_absolute_error(y_true, y_pred),
        'mse': mse,
        'rmse': np.sqrt(mse),
        'r2': 1 - (np.sum((y_true - y_pred)**2) / np.sum((y_true - y_true.mean())**2))
    }
    return metrics


def create_comparison_table(models_dict: dict, metrics_dict: dict) -> pd.DataFrame:
    """
    Crea tabla comparativa de métricas de modelos.
    
    Args:
        models_dict (dict): Diccionario con nombres y modelos
        metrics_dict (dict): Diccionario con métricas por modelo
        
    Returns:
        pd.DataFrame: Tabla comparativa
    """
    comparison_df = pd.DataFrame(metrics_dict).T
    return comparison_df


def save_metrics_table(df: pd.DataFrame, filepath: str) -> None:
    """
    Guarda tabla de métricas en CSV.
    
    Args:
        df (pd.DataFrame): Tabla de métricas
        filepath (str): Ruta de guardado
    """
    df.to_csv(filepath)
    print(f"✅ Tabla de métricas guardada en: {filepath}")


def plot_model_comparison(metrics_dict: dict, output_path: str = None) -> None:
    """
    Crea visualización comparativa de modelos.
    
    Args:
        metrics_dict (dict): Diccionario con métricas
        output_path (str): Ruta para guardar gráfico
    """
    df = pd.DataFrame(metrics_dict).T
    
    # Seleccionar solo las métricas principales
    plot_cols = [col for col in ['accuracy', 'precision', 'recall', 'f1'] if col in df.columns]
    
    if len(plot_cols) == 0:
        print("No hay métricas para visualizar")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Comparativa de Modelos', fontsize=16, fontweight='bold')
    
    # Accuracy
    if 'accuracy' in df.columns:
        df['accuracy'].plot(kind='bar', ax=axes[0, 0], color='skyblue')
        axes[0, 0].set_title('Accuracy')
        axes[0, 0].set_ylabel('Score')
        axes[0, 0].set_ylim([0, 1])
    
    # Precision
    if 'precision' in df.columns:
        df['precision'].plot(kind='bar', ax=axes[0, 1], color='lightcoral')
        axes[0, 1].set_title('Precision')
        axes[0, 1].set_ylabel('Score')
        axes[0, 1].set_ylim([0, 1])
    
    # Recall
    if 'recall' in df.columns:
        df['recall'].plot(kind='bar', ax=axes[1, 0], color='lightgreen')
        axes[1, 0].set_title('Recall')
        axes[1, 0].set_ylabel('Score')
        axes[1, 0].set_ylim([0, 1])
    
    # F1
    if 'f1' in df.columns:
        df['f1'].plot(kind='bar', ax=axes[1, 1], color='plum')
        axes[1, 1].set_title('F1-Score')
        axes[1, 1].set_ylabel('Score')
        axes[1, 1].set_ylim([0, 1])
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✅ Gráfico guardado en: {output_path}")
    plt.show()


def plot_confusion_matrix(y_true, y_pred, model_name: str = 'Modelo', output_path: str = None) -> None:
    """
    Crea matriz de confusión.
    
    Args:
        y_true: Target real
        y_pred: Predicciones
        model_name (str): Nombre del modelo
        output_path (str): Ruta para guardar
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True)
    plt.title(f'Matriz de Confusión - {model_name}')
    plt.ylabel('Real')
    plt.xlabel('Predicción')
    plt.xticks(rotation=45)
    plt.yticks(rotation=45)
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✅ Matriz de confusión guardada en: {output_path}")
    plt.show()


def get_feature_importances(model, feature_names: list = None):
    """
    Extrae importancia de features del modelo (si la soporta).
    
    Args:
        model: Modelo entrenado
        feature_names (list): Nombres de features
        
    Returns:
        pd.DataFrame: DataFrame con importancias
    """
    try:
        importances = model.named_steps['model'].feature_importances_
        if feature_names:
            return pd.DataFrame({'feature': feature_names, 'importance': importances}).sort_values('importance', ascending=False)
        else:
            return pd.Series(importances)
    except AttributeError:
        print("El modelo no tiene feature_importances_")
        return None


def plot_feature_importances(model, feature_names: list, top_n: int = 15, output_path: str = None) -> None:
    """
    Visualiza importancia de features.
    
    Args:
        model: Modelo entrenado
        feature_names (list): Nombres de features
        top_n (int): Top N features a mostrar
        output_path (str): Ruta para guardar
    """
    importances_df = get_feature_importances(model, feature_names)
    
    if importances_df is None:
        return
    
    top_features = importances_df.head(top_n)
    
    plt.figure(figsize=(10, 6))
    plt.barh(top_features['feature'], top_features['importance'])
    plt.xlabel('Importancia')
    plt.title(f'Top {top_n} Features por Importancia')
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✅ Gráfico de importancia guardado en: {output_path}")
    plt.show()


def print_classification_report(y_true, y_pred, target_names: list = None) -> None:
    """
    Imprime reporte de clasificación detallado.
    
    Args:
        y_true: Target real
        y_pred: Predicciones
        target_names (list): Nombres de las clases
    """
    report = classification_report(y_true, y_pred, target_names=target_names)
    print("\n" + "="*60)
    print("REPORTE DE CLASIFICACIÓN")
    print("="*60)
    print(report)
