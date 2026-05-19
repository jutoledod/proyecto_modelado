"""
hyperparameter_tuning.py

Módulo para optimización de hiperparámetros.
Incluye: GridSearchCV, RandomizedSearchCV, análisis de impacto, reportes.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import matplotlib.pyplot as plt
import seaborn as sns
import joblib


def grid_search_optimization(model, X_train, y_train, param_grid: dict, cv: int = 5, 
                             scoring: str = 'accuracy', n_jobs: int = -1) -> GridSearchCV:
    """
    Realiza búsqueda exhaustiva de hiperparámetros.
    
    Args:
        model: Modelo base (sin entrenar)
        X_train (pd.DataFrame): Features de entrenamiento
        y_train (pd.Series): Target de entrenamiento
        param_grid (dict): Grid de parámetros a probar
        cv (int): Número de folds
        scoring (str): Métrica para optimización
        n_jobs (int): Número de jobs (-1 = usar todos)
        
    Returns:
        GridSearchCV: Objeto con resultados de búsqueda
    """
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=cv,
        scoring=scoring,
        n_jobs=n_jobs,
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    return grid_search


def random_search_optimization(model, X_train, y_train, param_distributions: dict, 
                               n_iter: int = 20, cv: int = 5, scoring: str = 'accuracy', 
                               random_state: int = 42, n_jobs: int = -1) -> RandomizedSearchCV:
    """
    Realiza búsqueda aleatoria de hiperparámetros.
    
    Args:
        model: Modelo base (sin entrenar)
        X_train (pd.DataFrame): Features de entrenamiento
        y_train (pd.Series): Target de entrenamiento
        param_distributions (dict): Distribuciones de parámetros a probar
        n_iter (int): Número de iteraciones
        cv (int): Número de folds
        scoring (str): Métrica para optimización
        random_state (int): Seed para reproducibilidad
        n_jobs (int): Número de jobs (-1 = usar todos)
        
    Returns:
        RandomizedSearchCV: Objeto con resultados de búsqueda
    """
    random_search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_distributions,
        n_iter=n_iter,
        cv=cv,
        scoring=scoring,
        n_jobs=n_jobs,
        random_state=random_state,
        verbose=1
    )
    
    random_search.fit(X_train, y_train)
    return random_search


def get_best_params(search_result) -> dict:
    """
    Extrae los mejores parámetros de una búsqueda.
    
    Args:
        search_result: Resultado de GridSearchCV o RandomizedSearchCV
        
    Returns:
        dict: Diccionario con mejores parámetros
    """
    return search_result.best_params_


def get_best_score(search_result) -> float:
    """
    Extrae el mejor score de una búsqueda.
    
    Args:
        search_result: Resultado de GridSearchCV o RandomizedSearchCV
        
    Returns:
        float: Mejor score obtenido
    """
    return search_result.best_score_


def get_best_model(search_result):
    """
    Obtiene el mejor modelo de una búsqueda.
    
    Args:
        search_result: Resultado de GridSearchCV o RandomizedSearchCV
        
    Returns:
        Pipeline: Mejor modelo
    """
    return search_result.best_estimator_


def save_search_results(search_result, filepath: str) -> None:
    """
    Guarda resultados de búsqueda.
    
    Args:
        search_result: Resultado de búsqueda
        filepath (str): Ruta de guardado
    """
    joblib.dump(search_result, filepath)
    print(f"✅ Resultados de búsqueda guardados en: {filepath}")


def load_search_results(filepath: str):
    """
    Carga resultados de búsqueda guardados.
    
    Args:
        filepath (str): Ruta del archivo
        
    Returns:
        GridSearchCV o RandomizedSearchCV: Resultados cargados
    """
    return joblib.load(filepath)


def create_optimization_report(original_score: float, optimized_score: float, 
                               original_params: dict = None, optimized_params: dict = None) -> dict:
    """
    Crea reporte de optimización.
    
    Args:
        original_score (float): Score original
        optimized_score (float): Score después de optimización
        original_params (dict): Parámetros originales
        optimized_params (dict): Parámetros optimizados
        
    Returns:
        dict: Reporte con cambios y mejoras
    """
    improvement = ((optimized_score - original_score) / abs(original_score) * 100) if original_score != 0 else 0
    
    report = {
        'original_score': original_score,
        'optimized_score': optimized_score,
        'improvement_percentage': improvement,
        'original_params': original_params or {},
        'optimized_params': optimized_params or {},
    }
    
    if original_params and optimized_params:
        report['changed_params'] = {k: v for k, v in optimized_params.items() 
                                     if k in original_params and original_params[k] != v}
    
    return report


def plot_optimization_impact(reports_dict: dict, output_path: str = None) -> None:
    """
    Visualiza impacto de optimización en modelos.
    
    Args:
        reports_dict (dict): Diccionario con reportes de optimización
        output_path (str): Ruta para guardar gráfico
    """
    models = list(reports_dict.keys())
    original_scores = [reports_dict[m]['original_score'] for m in models]
    optimized_scores = [reports_dict[m]['optimized_score'] for m in models]
    improvements = [reports_dict[m]['improvement_percentage'] for m in models]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Gráfico 1: Comparación Original vs Optimizado
    x = np.arange(len(models))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, original_scores, width, label='Original', color='skyblue')
    bars2 = ax1.bar(x + width/2, optimized_scores, width, label='Optimizado', color='lightcoral')
    
    ax1.set_xlabel('Modelo', fontweight='bold')
    ax1.set_ylabel('Score', fontweight='bold')
    ax1.set_title('Comparación de Scores: Original vs Optimizado')
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # Gráfico 2: Mejora Porcentual
    colors = ['green' if imp > 0 else 'red' for imp in improvements]
    bars3 = ax2.bar(models, improvements, color=colors, alpha=0.7)
    ax2.set_xlabel('Modelo', fontweight='bold')
    ax2.set_ylabel('Mejora (%)', fontweight='bold')
    ax2.set_title('Mejora Porcentual por Optimización')
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax2.set_xticklabels(models, rotation=45, ha='right')
    ax2.grid(axis='y', alpha=0.3)
    
    # Agregar valores en barras
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}',
                    ha='center', va='bottom', fontsize=8)
    
    for bar, imp in zip(bars3, improvements):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{imp:.1f}%',
                ha='center', va='bottom' if imp > 0 else 'top', fontsize=8)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✅ Gráfico de optimización guardado en: {output_path}")
    plt.show()


def save_optimization_report(reports_dict: dict, filepath: str) -> None:
    """
    Guarda reporte de optimización en CSV.
    
    Args:
        reports_dict (dict): Diccionario con reportes
        filepath (str): Ruta de guardado
    """
    report_data = []
    
    for model_name, report in reports_dict.items():
        report_data.append({
            'modelo': model_name,
            'score_original': report['original_score'],
            'score_optimizado': report['optimized_score'],
            'mejora_porcentaje': report['improvement_percentage']
        })
    
    df_report = pd.DataFrame(report_data)
    df_report.to_csv(filepath, index=False)
    print(f"✅ Reporte de optimización guardado en: {filepath}")


def print_optimization_summary(reports_dict: dict) -> None:
    """
    Imprime resumen de optimización.
    
    Args:
        reports_dict (dict): Diccionario con reportes
    """
    print("\n" + "="*70)
    print("RESUMEN DE OPTIMIZACIÓN DE HIPERPARÁMETROS")
    print("="*70)
    
    for model_name, report in reports_dict.items():
        print(f"\n{model_name}:")
        print(f"  Score Original:    {report['original_score']:.4f}")
        print(f"  Score Optimizado:  {report['optimized_score']:.4f}")
        print(f"  Mejora:            {report['improvement_percentage']:+.2f}%")
    
    print("\n" + "="*70)
    
    df = pd.DataFrame(report_data)
    df.to_csv(filepath, index=False)
