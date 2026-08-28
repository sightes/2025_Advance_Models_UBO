# 2025_Advance_Models_UBO

[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?logo=python)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter)](https://jupyter.org/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3+-F7931E?logo=scikit-learn)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/Pandas-2+-15045D?logo=pandas)](https://pandas.pydata.org/)
[![License](https://img.shields.io/badge/License-Private-red)](#licencia)
[![Release](https://img.shields.io/badge/Release-v1.0.0-green)](https://github.com/sightes/2025_Advance_Models_UBO/releases/tag/v1.0.0)

Repositorio de la asignatura Modelos Avanzados de la Universidad Bernardo O'Higgins (UBO) para el año 2025.

---

## About

2025_Advance_Models_UBO es un repositorio que contiene materiales de curso, proyectos de laboratorio, tareas y trabajos finales de la asignatura Modelos Avanzados de la UBO. El enfoque principal es el aprendizaje automático (Machine Learning), incluyendo técnicas de clustering, regresión, clasificación y análisis de valores extremos.

**¿Por qué este repositorio?**

- **Contenido Académico**: Material completo de la asignatura Modelos Avanzados
- **Proyectos Prácticos**: Notebooks de Jupyter con implementaciones reales
- **Múltiples Técnicas**: K-Means, regresión, clasificación, valores extremos
- **Documentación Detallada**: Cada clase y tarea tiene su propia carpeta
- **Fácil Acceso**: Estructura organizada por semanas y temas
- **Reutilizable**: Código listo para usar y modificar

Ideal para estudiantes de ciencia de datos, machine learning, o cualquier persona que quiera aprender técnicas avanzadas de modelado estadístico.

---

## Características principales

| Característica | Estado | Descripción |
|----------------|--------|-------------|
| Notebooks Jupyter | ✅ | Código interactivo y visualizaciones |
| Clustering K-Means | ✅ | Implementación completa con método del codo |
| Regresión | ✅ | Modelos de regresión lineal y múltiple |
| Clasificación | ✅ | Algoritmos de clasificación supervisada |
| Valores Extremos | ✅ | Análisis de outliers y valores atípicos |
| Visualizaciones | ✅ | Gráficos con Matplotlib y Seaborn |
| Preprocesamiento | ✅ | Normalización y escalado de datos |
| Tareas Prácticas | ✅ | Ejercicios para practicar conceptos |
| Trabajo Final | ✅ | Proyectos integradores de conocimientos |
| Makefile Universal | ✅ | Comandos para crear estructura en cualquier lenguaje |

---

## Estructura del proyecto

```
2025_Advance_Models_UBO/
├── clase-22-03-2025-ML/           # Clase de Machine Learning
├── Clase_25_03_2025/              # Clase del 25 de marzo
├── clase_25_04_ML/                # Clase del 25 de abril
├── Semana 5 abril - Machine Learning.ipynb  # Notebook semana 5
├── KMEANS_V3.ipynb                # Implementación K-Means v3
├── ejemplo-kmeans.ipynb           # Ejemplo de K-Means
├── ejemploml.py                   # Script de ejemplo ML
├── Tarea N1 ML 04-04/            # Tarea 1 de ML
├── Tarea N2 ML 24-04/            # Tarea 2 de ML
├── Tarea_electivo_profundizacion_1/  # Tarea de electivo
├── Introduccion_Valores_Extremos/    # Introducción a valores extremos
├── Actividad_Final_Valores_Extremos/ # Actividad final valores extremos
├── Trabajo final DL/             # Trabajo final Deep Learning
├── trabajo final ML/             # Trabajo final Machine Learning
├── Makefile                      # Comandos para crear estructura
├── .pre-commit-config.yaml       # Configuración de pre-commit
├── .sqlfluff                     # Configuración de SQL linting
├── .gitignore                    # Archivos ignorados
├── grafica.png                   # Gráfica generada
├── grafico.png                   # Gráfico generado
└── output.png                    # Salida generada
```

---

## Contenido del curso

### Semana 1-2: Introducción al Machine Learning
- Conceptos básicos de ML
- Tipos de aprendizaje (supervisado, no supervisado)
- Preprocesamiento de datos

### Semana 3-4: Clustering con K-Means
- Algoritmo K-Means
- Método del codo para determinar k
- Interpretación de resultados

### Semana 5: Regresión
- Regresión lineal simple y múltiple
- Métricas de evaluación (R², MSE, RMSE)
- Validación de modelos

### Semana 6-7: Clasificación
- Algoritmos de clasificación
- Matriz de confusión
- Precisión, recall, F1-score

### Semana 8: Valores Extremos
- Detección de outliers
- Métodos estadísticos
- Aplicaciones prácticas

### Trabajo Final
- Proyecto integrador
- Aplicación de todas las técnicas aprendidas
- Presentación de resultados

---

## Tecnologías utilizadas

| Categoría | Tecnologías |
|-----------|-------------|
| Lenguaje | Python 3.9+ |
| Notebooks | Jupyter Notebook |
| Machine Learning | Scikit-learn, XGBoost |
| Manipulación de datos | Pandas, NumPy |
| Visualización | Matplotlib, Seaborn |
| Preprocesamiento | Scikit-learn (StandardScaler, PCA) |
| Clustering | K-Means (Scikit-learn) |
| Regresión | Scikit-learn (LinearRegression) |
| Clasificación | Scikit-learn (RandomForest, SVM) |

---

## Instalación

### Requisitos
- Python 3.9+
- pip

### Instalar dependencias
```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

### Ejecutar notebooks
```bash
jupyter notebook
```

### Crear nuevo proyecto
```bash
# Python
make PROJECT="mi_nuevo_proyecto" LANGUAGE="Python" all
```

---

## Uso

### Ejecutar K-Means
```python
from sklearn.cluster import KMeans
import pandas as pd

# Cargar datos
df = pd.read_csv('datos.csv')

# Seleccionar features
features = ['feature1', 'feature2', 'feature3']
X = df[features]

# Aplicar K-Means
kmeans = KMeans(n_clusters=3, random_state=42)
df['cluster'] = kmeans.fit_predict(X)
```

### Ejecutar regresión
```python
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Dividir datos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Entrenar modelo
model = LinearRegression()
model.fit(X_train, y_train)

# Predecir
predictions = model.predict(X_test)
```

---

## Contribuir

Las contribuciones son bienvenidas. Por favor, sigue estos pasos:

### 1. Fork el repositorio
```bash
git fork https://github.com/sightes/2025_Advance_Models_UBO.git
```

### 2. Clona tu fork
```bash
git clone https://github.com/tu-usuario/2025_Advance_Models_UBO.git
cd 2025_Advance_Models_UBO
```

### 3. Crea una rama para tu feature
```bash
git checkout -b feature/nueva-funcionalidad
```

### 4. Haz tus cambios
- Sigue la guía de estilo del proyecto
- Agrega documentación para nuevos conceptos
- Asegúrate de que el código funcione correctamente

### 5. Commit tus cambios
```bash
git commit -m "Add: descripción de tu cambio"
```

### 6. Push a tu fork
```bash
git push origin feature/nueva-funcionalidad
```

### 7. Crea un Pull Request
- Ve al repositorio original
- Haz clic en "New Pull Request"
- Selecciona tu rama
- Agrega una descripción clara de los cambios

### Convenciones de commits

- `Add:` para nuevos notebooks o código
- `Fix:` para corrección de bugs
- `Update:` para mejoras
- `Remove:` para eliminaciones
- `Docs:` para documentación

### Estructura de un nuevo notebook

Cada notebook debe incluir:
1. Título descriptivo
2. Objetivo del notebook
3. Importación de librerías
4. Carga y exploración de datos
5. Preprocesamiento
6. Implementación del modelo
7. Evaluación de resultados
8. Conclusiones

### Reporte de bugs

Si encuentras un bug, por favor abre un issue con:
1. Título descriptivo
2. Pasos para reproducir
3. Comportamiento esperado vs actual
4. Capturas de pantalla si aplica
5. Información del sistema (SO, versión de Python)

---

## Licencia

Privado — Proyecto de demostración.

---

Mantenido con ☕ y código diario.