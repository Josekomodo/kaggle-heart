# Implementation Plan: Heart Disease EDA

## Overview

Script Python standalone `eda.py` que ejecuta un EDA completo sobre los datos de Kaggle Playground Series S6E2. Las tareas siguen el flujo lineal del diseño: carga → análisis → visualizaciones → conclusiones. Los tests de propiedades usan Hypothesis con `max_examples=100`.

## Tasks

- [x] 1. Estructura base y carga de datos
  - Crear `eda.py` con las constantes globales (`TRAIN_PATH`, `TEST_PATH`, `OUTPUT_DIR`, `RANDOM_STATE`, `COL_RENAME`, `NUMERIC_FEATURES`, `CATEGORICAL_FEATURES`)
  - Implementar `load_and_validate(train_path, test_path)`: carga CSVs, renombra columnas a snake_case via `COL_RENAME`, codifica target (`Presence=1`, `Absence=0`), imprime shape/dtypes/head(5)/nulos/duplicados
  - Crear `eda_outputs/` con `os.makedirs(OUTPUT_DIR, exist_ok=True)` al inicio de `main()`
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5_

  - [x] 1.1 Write property test — Property 1: codificación binaria del target
    - **Property 1: target encoding preserves counts**
    - Generar listas aleatorias de `"Presence"/"Absence"` con Hypothesis, verificar que `target` ∈ {0,1} y que `sum(target==1) == labels.count("Presence")`
    - `@settings(max_examples=100)`
    - **Validates: Requirements 1.5**

- [x] 2. Análisis del target y features numéricas
  - Implementar `analyze_target(train_df)`: conteo/porcentaje por clase, genera `01_target_distribution.png`, retorna dict con `counts`, `pct_minority`, `is_imbalanced`
  - Implementar `analyze_numeric_features(train_df)`: estadísticas descriptivas (media, mediana, std, min, max, p25, p75), detecta ceros anómalos en `bp`/`cholesterol`, genera `02_numeric_distributions.png` (histogramas KDE por clase) y `03_numeric_boxplots.png` (boxplots por clase)
  - _Requirements: 2.1, 2.2, 2.3, 3.1, 3.2, 3.3, 3.4_

  - [x] 2.1 Write property test — Property 3: detección de ceros anómalos
    - **Property 3: zero anomaly detection**
    - Generar DataFrames con Hypothesis donde `bp` o `cholesterol` contengan al menos un 0, verificar que `findings["numeric"]["zero_counts"]` reporta conteo > 0 para esa feature
    - `@settings(max_examples=100)`
    - **Validates: Requirements 3.4**

- [x] 3. Análisis de features categóricas y correlaciones
  - Implementar `analyze_categorical_features(train_df)`: tasa de `Presence` por valor único, test chi-cuadrado por feature, genera `04_categorical_vs_target.png` (barras agrupadas/apiladas)
  - Implementar `analyze_correlations(train_df)`: matriz Pearson sobre numéricas + target, correlación Spearman feature-target ordenada por |valor| desc, genera `05_correlation_heatmap.png` y `06_spearman_correlations.png`
  - _Requirements: 4.1, 4.2, 4.3, 5.1, 5.2, 5.3, 5.4_

  - [x] 3.1 Write property test — Property 6: ordenación de correlaciones Spearman
    - **Property 6: Spearman correlations sorted by absolute value descending**
    - Generar DataFrames aleatorios con Hypothesis, verificar que `findings["correlations"]["spearman_target"]` está ordenada de mayor a menor por valor absoluto
    - `@settings(max_examples=100)`
    - **Validates: Requirements 5.3, 10.2**

- [x] 4. Análisis train vs test y outliers
  - Implementar `analyze_train_test_distribution(train_df, test_df)`: superpone KDE de cada feature numérica train/test, test KS por feature, marca features con p-value < 0.05 como problemáticas, genera `07_train_test_distribution.png`
  - Implementar `analyze_outliers(train_df)`: criterio IQR `[Q1 - 1.5*IQR, Q3 + 1.5*IQR]`, reporta conteo y porcentaje por feature, genera `08_violin_plots.png`
  - _Requirements: 6.1, 6.2, 6.3, 7.1, 7.2, 7.3_

  - [x] 4.1 Write property test — Property 5: marcado de features con drift KS
    - **Property 5: KS drift flagging**
    - Generar pares de distribuciones con Hypothesis donde el test KS resulte en p-value < 0.05, verificar que la feature aparece en `findings["train_test"]["problematic_features"]`
    - `@settings(max_examples=100)`
    - **Validates: Requirements 6.3**

  - [x] 4.2 Write property test — Property 4: consistencia de outliers IQR
    - **Property 4: IQR outlier count consistency**
    - Generar arrays numéricos con Hypothesis, verificar que el conteo reportado coincide exactamente con el número de valores fuera de `[Q1 - 1.5*IQR, Q3 + 1.5*IQR]`
    - `@settings(max_examples=100)`
    - **Validates: Requirements 7.1, 7.2**

- [x] 5. Checkpoint — Verificar tests unitarios y de propiedades
  - Asegurarse de que todos los tests pasan hasta este punto. Preguntar al usuario si hay dudas antes de continuar.

- [x] 6. Análisis de interacciones y feature importance
  - Implementar `analyze_interactions(train_df)`: pairplot numéricas por target (`09_pairplot.png`), scatter `age` vs `max_hr` (`10_age_vs_maxhr.png`), scatter `st_depression` vs `max_hr` (`11_stdep_vs_maxhr.png`), calcula correlación Spearman entre todos los pares numéricos, selecciona top 3 excluyendo los pares fijos, genera `12_top_pair_1.png`, `12_top_pair_2.png`, `12_top_pair_3.png`
  - Implementar `compute_feature_importance(train_df)`: entrena `RandomForestClassifier(random_state=42)`, extrae importancias ordenadas desc, calcula ROC AUC con CV 5-fold, genera `13_feature_importance_rf.png`
  - _Requirements: 8.1, 8.2, 8.3, 8.4, 9.1, 9.2, 9.3_

  - [x] 6.1 Write property test — Property 7: selección de top pares por correlación
    - **Property 7: top pairs selection by absolute Spearman correlation**
    - Generar DataFrames con Hypothesis con al menos 5 features numéricas, verificar que los 3 pares seleccionados tienen correlación absoluta ≥ que cualquier otro par elegible (excluyendo pares fijos)
    - `@settings(max_examples=100)`
    - **Validates: Requirements 8.4**

- [x] 7. Generación de conclusiones y wiring final
  - Implementar `write_conclusions(output_dir, findings)`: escribe `eda_outputs/conclusions.md` con secciones: Overview, Class Balance, Numeric Features, Categorical Features, Correlations, Train-Test Shift, Outliers, Feature Importance, Recommendations. Incluye top 3 features Spearman, nota de desbalance si aplica, ceros anómalos, features problemáticas KS, ROC AUC CV, recomendaciones de preprocesamiento
  - Implementar `print_summary()`: imprime en consola la ruta de `OUTPUT_DIR` y el número total de PNGs generados
  - Cablear `main()`: llamar todas las funciones en orden, pasar hallazgos a `write_conclusions`, manejar errores por función con try/except individual
  - _Requirements: 2.3, 3.4, 6.3, 7.2, 9.3, 10.1, 10.2, 10.3, 10.4_

- [x] 8. Tests unitarios en `tests/test_eda.py`
  - [x] 8.1 Write unit tests — carga y codificación
    - `test_target_encoding`: verifica `Presence`→1 y `Absence`→0 con fixture sintético
    - `test_zero_anomaly_detection`: verifica que `bp=0` y `cholesterol=0` se reportan en `zero_counts`
    - _Requirements: 1.5, 3.4_

  - [x] 8.2 Write unit tests — estadísticos
    - `test_ks_flagging`: verifica que features con p-value KS < 0.05 aparecen en `problematic_features`
    - `test_outlier_count_iqr`: verifica conteo IQR correcto con datos conocidos
    - `test_spearman_ordering`: verifica que la serie está ordenada por |valor| desc
    - _Requirements: 6.3, 7.1, 5.3_

  - [x] 8.3 Write integration test — conteo de imágenes
    - `test_output_image_count`: ejecuta el script completo sobre datos reales y verifica exactamente 15 PNGs en `eda_outputs/` y existencia de `conclusions.md`
    - **Property 2: image count completeness**
    - **Validates: Requirements 10.4**
    - _Requirements: 10.4_

- [ ] 9. Checkpoint final — Todos los tests pasan
  - Ejecutar `pytest tests/test_eda.py -v` y verificar que todos los tests pasan. Preguntar al usuario si hay dudas antes de cerrar.

## Notes

- Las sub-tareas marcadas con `*` son opcionales y pueden omitirse para un MVP más rápido
- Cada tarea referencia los requirements específicos para trazabilidad
- Property 2 se verifica con test de integración (no PBT) por ser una verificación de I/O determinista
- Todos los PBT usan `@settings(max_examples=100)` y el tag `# Feature: heart-disease-eda, Property N`
- `random_state=42` en todos los modelos y operaciones estocásticas
