# Requirements Document

## Introduction

EDA (Exploratory Data Analysis) completo para la competición Kaggle Playground Series S6E2 — Heart Disease Prediction. El objetivo es analizar en profundidad los datos clínicos de 13 features para entender la distribución de las variables, su relación con el target `Heart Disease` (Presence/Absence), detectar anomalías o patrones relevantes, y extraer conclusiones accionables para el modelado posterior. El análisis produce visualizaciones guardadas como imágenes y un resumen de conclusiones documentado.

## Glossary

- **EDA_Script**: Script Python (`eda.py`) que ejecuta el análisis exploratorio completo de forma reproducible.
- **Train_Dataset**: Archivo `playground-series-s6e2/train.csv` con ~630.000 filas y 15 columnas (id, 13 features, target).
- **Test_Dataset**: Archivo `playground-series-s6e2/test.csv` con los mismos 13 features sin target.
- **Target**: Columna `Heart Disease` con valores `Presence` (1) y `Absence` (0).
- **Numeric_Features**: Features continuas o discretas numéricas: `Age`, `BP`, `Cholesterol`, `Max HR`, `ST depression`.
- **Categorical_Features**: Features con valores discretos codificados: `Sex`, `Chest pain type`, `FBS over 120`, `EKG results`, `Exercise angina`, `Slope of ST`, `Number of vessels fluro`, `Thallium`.
- **Output_Dir**: Directorio `eda_outputs/` donde se guardan todas las imágenes generadas.
- **Conclusions_File**: Archivo `eda_outputs/conclusions.md` con las conclusiones textuales del análisis.

---

## Requirements

### Requirement 1: Carga y validación inicial de datos

**User Story:** As a data scientist, I want to load and validate the datasets, so that I can confirm data integrity before analysis.

#### Acceptance Criteria

1. THE EDA_Script SHALL load `Train_Dataset` y `Test_Dataset` usando `pandas.read_csv`.
2. THE EDA_Script SHALL imprimir el shape, dtypes y las primeras 5 filas de cada dataset.
3. THE EDA_Script SHALL reportar el número de valores nulos por columna en `Train_Dataset` y `Test_Dataset`.
4. THE EDA_Script SHALL reportar el número de filas duplicadas en `Train_Dataset`.
5. WHEN `Train_Dataset` es cargado, THE EDA_Script SHALL codificar `Heart Disease` como entero binario (`Presence=1`, `Absence=0`) antes de cualquier análisis estadístico.

---

### Requirement 2: Análisis del target (class balance)

**User Story:** As a data scientist, I want to understand the class distribution of the target variable, so that I can assess class imbalance and its impact on modeling.

#### Acceptance Criteria

1. THE EDA_Script SHALL calcular y mostrar el conteo absoluto y el porcentaje de cada clase (`Presence`, `Absence`) en `Train_Dataset`.
2. THE EDA_Script SHALL generar un gráfico de barras con la distribución del `Target` y guardarlo como `Output_Dir/01_target_distribution.png`.
3. IF la clase minoritaria representa menos del 40% del total, THEN THE EDA_Script SHALL incluir una nota en `Conclusions_File` indicando el desbalance y su relevancia para ROC AUC.

---

### Requirement 3: Estadísticas descriptivas de features numéricas

**User Story:** As a data scientist, I want descriptive statistics for numeric features, so that I can identify outliers, skewness, and scale differences.

#### Acceptance Criteria

1. THE EDA_Script SHALL calcular media, mediana, desviación estándar, mínimo, máximo y percentiles (25, 75) para cada feature en `Numeric_Features`.
2. THE EDA_Script SHALL generar histogramas con curva KDE para cada feature en `Numeric_Features`, separados por clase del `Target`, y guardarlos como `Output_Dir/02_numeric_distributions.png`.
3. THE EDA_Script SHALL generar boxplots de cada feature en `Numeric_Features` agrupados por `Target` y guardarlos como `Output_Dir/03_numeric_boxplots.png`.
4. WHEN una feature en `Numeric_Features` tiene valores iguales a 0 que sean clínicamente anómalos (e.g., `Cholesterol=0`, `BP=0`), THE EDA_Script SHALL reportar el conteo de dichos valores en `Conclusions_File`.

---

### Requirement 4: Análisis de features categóricas

**User Story:** As a data scientist, I want to analyze categorical features against the target, so that I can identify which categories are most predictive of heart disease.

#### Acceptance Criteria

1. THE EDA_Script SHALL calcular la tasa de `Presence` (Heart Disease rate) por cada valor único de cada feature en `Categorical_Features`.
2. THE EDA_Script SHALL generar gráficos de barras apiladas o agrupadas mostrando la distribución de `Target` por cada feature en `Categorical_Features` y guardarlos como `Output_Dir/04_categorical_vs_target.png`.
3. THE EDA_Script SHALL calcular el test chi-cuadrado entre cada feature en `Categorical_Features` y el `Target`, reportando el p-value en `Conclusions_File`.

---

### Requirement 5: Análisis de correlaciones

**User Story:** As a data scientist, I want to analyze feature correlations, so that I can detect multicollinearity and identify the most linearly related features to the target.

#### Acceptance Criteria

1. THE EDA_Script SHALL calcular la matriz de correlación de Pearson sobre todas las features numéricas más el `Target` codificado.
2. THE EDA_Script SHALL generar un heatmap de la matriz de correlación y guardarlo como `Output_Dir/05_correlation_heatmap.png`.
3. THE EDA_Script SHALL calcular la correlación de Spearman entre cada feature (numéricas y categóricas) y el `Target` codificado, ordenada por valor absoluto descendente.
4. THE EDA_Script SHALL generar un gráfico de barras horizontales con las correlaciones de Spearman feature-target y guardarlo como `Output_Dir/06_spearman_correlations.png`.

---

### Requirement 6: Análisis de distribución train vs test

**User Story:** As a data scientist, I want to compare train and test feature distributions, so that I can detect distribution shift that could degrade model performance.

#### Acceptance Criteria

1. THE EDA_Script SHALL superponer las distribuciones (KDE o histograma normalizado) de cada feature en `Numeric_Features` para `Train_Dataset` y `Test_Dataset` en un mismo gráfico y guardarlo como `Output_Dir/07_train_test_distribution.png`.
2. THE EDA_Script SHALL calcular el estadístico KS (Kolmogorov-Smirnov) entre train y test para cada feature en `Numeric_Features` y reportar los resultados en `Conclusions_File`.
3. WHEN el p-value del test KS para una feature es menor a 0.05, THE EDA_Script SHALL marcar dicha feature como potencialmente problemática en `Conclusions_File`.

---

### Requirement 7: Análisis de outliers

**User Story:** As a data scientist, I want to detect outliers in numeric features, so that I can decide whether to cap, transform, or remove them before modeling.

#### Acceptance Criteria

1. THE EDA_Script SHALL identificar outliers en `Numeric_Features` usando el criterio IQR (valores fuera de [Q1 - 1.5*IQR, Q3 + 1.5*IQR]).
2. THE EDA_Script SHALL reportar el conteo y porcentaje de outliers por feature en `Conclusions_File`.
3. THE EDA_Script SHALL generar un gráfico de violín para cada feature en `Numeric_Features` agrupado por `Target` y guardarlo como `Output_Dir/08_violin_plots.png`.

---

### Requirement 8: Análisis de interacciones entre features

**User Story:** As a data scientist, I want to explore pairwise feature interactions, so that I can identify non-linear relationships and feature combinations relevant to the target.

#### Acceptance Criteria

1. THE EDA_Script SHALL generar un pairplot de las features en `Numeric_Features` coloreado por `Target` y guardarlo como `Output_Dir/09_pairplot.png`.
2. THE EDA_Script SHALL generar un scatter plot de `Age` vs `Max HR` coloreado por `Target` y guardarlo como `Output_Dir/10_age_vs_maxhr.png`.
3. THE EDA_Script SHALL generar un scatter plot de `ST depression` vs `Max HR` coloreado por `Target` y guardarlo como `Output_Dir/11_stdep_vs_maxhr.png`.
4. THE EDA_Script SHALL calcular la correlación de Spearman entre todos los pares de features numéricas, identificar los 3 pares con mayor correlación absoluta que no sean (`Age`, `Max HR`) ni (`ST depression`, `Max HR`), y generar un scatter plot coloreado por `Target` para cada uno, guardándolos como `Output_Dir/12_top_pair_1.png`, `Output_Dir/12_top_pair_2.png` y `Output_Dir/12_top_pair_3.png`.

---

### Requirement 9: Feature importance preliminar

**User Story:** As a data scientist, I want a preliminary feature importance ranking, so that I can prioritize features for modeling and feature engineering.

#### Acceptance Criteria

1. THE EDA_Script SHALL entrenar un `RandomForestClassifier` con parámetros por defecto sobre `Train_Dataset` para obtener importancias de features.
2. THE EDA_Script SHALL generar un gráfico de barras horizontales con las importancias de features del `RandomForestClassifier` ordenadas descendentemente y guardarlo como `Output_Dir/13_feature_importance_rf.png`.
3. THE EDA_Script SHALL calcular el ROC AUC del `RandomForestClassifier` sobre `Train_Dataset` usando validación cruzada de 5 folds y reportar la media y desviación estándar en `Conclusions_File`.

---

### Requirement 10: Generación de conclusiones documentadas

**User Story:** As a data scientist, I want a written summary of EDA findings, so that I can quickly reference key insights when building models.

#### Acceptance Criteria

1. THE EDA_Script SHALL crear el archivo `Conclusions_File` con secciones estructuradas: Overview, Class Balance, Numeric Features, Categorical Features, Correlations, Train-Test Shift, Outliers, Feature Importance, y Recommendations.
2. THE EDA_Script SHALL incluir en `Conclusions_File` las 3 features con mayor correlación de Spearman con el `Target`.
3. THE EDA_Script SHALL incluir en `Conclusions_File` recomendaciones concretas de preprocesamiento basadas en los hallazgos del EDA (e.g., tratamiento de ceros anómalos, encoding de categóricas, escalado).
4. WHEN el EDA_Script termina su ejecución, THE EDA_Script SHALL imprimir en consola la ruta de `Output_Dir` y el número total de imágenes generadas.
