# Evaluación Parcial N°2: Modelado de Machine Learning

## 📋 Descripción del Proyecto

Este proyecto implementa un **ciclo completo de Machine Learning** con datos médicos, integrando:
- Modelos supervisados (clasificación/regresión)
- Modelos no supervisados (clustering, reducción de dimensionalidad)
- Evaluación rigurosa con validación cruzada
- Optimización de hiperparámetros
- Análisis comparativo detallado

## 🎯 Objetivos

1. Desarrollar múltiples modelos supervisados con Scikit-learn
2. Aplicar técnicas no supervisadas para exploración de datos
3. Realizar evaluación comparativa exhaustiva
4. Optimizar hiperparámetros usando GridSearchCV y RandomizedSearchCV
5. Documentar justificación técnica de todas las decisiones

## 📁 Estructura del Proyecto

```
proyecto_modelado/
├── notebooks/                          # Jupyter Notebooks del análisis
│   ├── 01_exploratory_analysis.ipynb   # Análisis exploratorio de datos
│   ├── 02_supervised_modeling.ipynb    # Modelos supervisados + no supervisados
│   ├── 03_model_evaluation.ipynb       # Evaluación comparativa
│   ├── 04_hyperparameter_optimization.ipynb  # Optimización de parámetros
│   └── 05_final_analysis.ipynb         # Análisis final e integración
├── src/                                # Código fuente modular
│   ├── data_preprocessing.py           # Funciones de preparación de datos
│   ├── model_training.py               # Entrenamiento de modelos
│   ├── model_evaluation.py             # Evaluación y comparación
│   └── hyperparameter_tuning.py        # Optimización de hiperparámetros
├── models/
│   └── trained_models/                 # Modelos serializados (.pkl, .joblib)
├── results/
│   ├── metrics/                        # Tablas de métricas (CSV)
│   ├── plots/                          # Visualizaciones (PNG, PDF)
│   └── reports/                        # Reportes generados
├── data/                               # Datos del proyecto
├── requirements.txt                    # Dependencias del proyecto
└── README.md                           # Este archivo
```

## 🚀 Instalación

### Requisitos Previos
- Python 3.8 o superior
- pip o conda

### Pasos de Instalación

1. **Clonar/descargar el proyecto:**
   ```bash
   cd c:\Users\juany\Desktop\raiz\proyecto_modelado
   ```

2. **Crear entorno virtual (recomendado):**
   ```bash
   python -m venv venv
   venv\Scripts\activate
   ```

3. **Instalar dependencias:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Iniciar Jupyter:**
   ```bash
   jupyter notebook
   ```

## 📊 Flujo de Trabajo

### Fase 0: Preparación ✅
- [x] Estructura de carpetas creada
- [x] Dependencias especificadas
- [x] Datos copiados

### Fase 1: Exploración (01_exploratory_analysis.ipynb)
- [ ] Cargar y validar datos
- [ ] Análisis descriptivo
- [ ] Visualizaciones
- [ ] Definir variable objetivo

### Fase 2: Preparación de Datos (data_preprocessing.py)
- [ ] Manejo de valores nulos
- [ ] Eliminación de duplicados
- [ ] Detección de outliers
- [ ] Encoding de variables
- [ ] Normalización/escalado

### Fase 3: Modelos Supervisados (02_supervised_modeling.ipynb + model_training.py)
- [ ] Implementar ≥4 modelos diferentes
- [ ] Crear pipelines de Scikit-learn
- [ ] Guardar modelos entrenados
- [ ] Documentar justificación

### Fase 4: Modelos No Supervisados (02_supervised_modeling.ipynb)
- [ ] Clustering (K-Means, Hierarchical, DBSCAN)
- [ ] Reducción de dimensionalidad (PCA)
- [ ] Análisis y visualización

### Fase 5: Evaluación Comparativa (03_model_evaluation.ipynb + model_evaluation.py)
- [ ] Cross-validation k-fold
- [ ] Calcular todas las métricas
- [ ] Tabla comparativa
- [ ] Visualizaciones avanzadas

### Fase 6: Optimización (04_hyperparameter_optimization.ipynb + hyperparameter_tuning.py)
- [ ] GridSearchCV
- [ ] RandomizedSearchCV
- [ ] Análisis de impacto
- [ ] Documentar mejoras

### Fase 7: Análisis Final (05_final_analysis.ipynb)
- [ ] Resumen ejecutivo
- [ ] Recomendaciones
- [ ] Limitaciones y reflexión

### Fase 8: Artefactos Finales
- [ ] Métricas en results/metrics/
- [ ] Visualizaciones en results/plots/
- [ ] Modelos en models/trained_models/

## 🔑 Aspectos Formales Importantes

- ✅ **Reproducibilidad 100%:** Uso de seeds y random_state en todos los modelos
- ✅ **Código limpio:** Docstrings en todas las funciones
- ✅ **Modularidad:** Funciones reutilizables en módulos separados
- ✅ **Manejo robusto:** Validación de entradas y manejo de excepciones
- ✅ **Justificación técnica:** Documentar decisiones en cada fase

## 📚 Dependencias Principales

- **pandas:** Manipulación de datos
- **numpy:** Operaciones numéricas
- **scikit-learn:** Machine Learning
- **matplotlib/seaborn:** Visualizaciones
- **jupyter:** Notebooks interactivos
- **xgboost/lightgbm:** Gradient Boosting (opcional)

## 📝 Notas

- Los datos están en la carpeta `data/`
- Los modelos se guardan en formato .joblib para facilitar deserialización
- Las visualizaciones se exportan en PNG a 300 DPI mínimo
- Cada notebook incluye markdown explicativo

## 🎓 Autor

Evaluación Parcial N°2 - SCY1101: Programación para la Ciencia de Datos
