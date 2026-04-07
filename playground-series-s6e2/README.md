# Kaggle Playground Series S6E2 — Predicting Heart Disease

## Descripción del problema

Competición de clasificación binaria cuyo objetivo es predecir la probabilidad de que un paciente tenga una enfermedad cardíaca (`Heart Disease`) a partir de datos clínicos. Los datos son sintéticos generados a partir del dataset original de Cleveland Heart Disease.

- Tipo: Clasificación binaria (probabilística)
- Métrica de evaluación: ROC AUC (Area Under the ROC Curve)
- Organizador: Kaggle
- Equipos participantes: ~4.370

---

## Archivos del dataset

| Archivo | Descripción |
|---|---|
| `train.csv` | Datos de entrenamiento con etiqueta `Heart Disease` |
| `test.csv` | Datos de test sin etiqueta (IDs desde 630000) |
| `sample_submission.csv` | Formato esperado para la entrega |

---

## Variables (features)

| Columna | Tipo | Descripción |
|---|---|---|
| `id` | int | Identificador único del paciente |
| `Age` | int | Edad en años |
| `Sex` | int | Sexo (1 = masculino, 0 = femenino) |
| `Chest pain type` | int | Tipo de dolor torácico (1–4) |
| `BP` | int | Presión arterial en reposo (mm Hg) |
| `Cholesterol` | int | Colesterol sérico (mg/dl) |
| `FBS over 120` | int | Glucemia en ayunas > 120 mg/dl (1 = sí, 0 = no) |
| `EKG results` | int | Resultados del electrocardiograma en reposo (0, 1, 2) |
| `Max HR` | int | Frecuencia cardíaca máxima alcanzada |
| `Exercise angina` | int | Angina inducida por ejercicio (1 = sí, 0 = no) |
| `ST depression` | float | Depresión del segmento ST inducida por ejercicio |
| `Slope of ST` | int | Pendiente del segmento ST en el pico del ejercicio (1–3) |
| `Number of vessels fluro` | int | Número de vasos principales coloreados por fluoroscopía (0–3) |
| `Thallium` | int | Resultado del test de talio (3 = normal, 6 = defecto fijo, 7 = defecto reversible) |

---

## Variable objetivo

| Columna | Valores en train | Descripción |
|---|---|---|
| `Heart Disease` | `Presence` / `Absence` | Diagnóstico de enfermedad cardíaca |

En la submission se predice como probabilidad continua entre 0 y 1.

---

## Evaluación

La métrica es el **ROC AUC** entre la probabilidad predicha y el target observado. Un valor de 0.5 equivale a predicción aleatoria y 1.0 es predicción perfecta.

```python
from sklearn.metrics import roc_auc_score
score = roc_auc_score(y_true, y_pred_proba)
```

---

## Formato de submission

El archivo debe tener cabecera y una fila por cada `id` del test set, con la probabilidad predicha de `Presence`:

```
id,Heart Disease
630000,0.85
630001,0.12
630002,0.67
...
```

El test set contiene IDs desde `630000` hasta `633xxx` (aprox. 3.000+ registros).

---

## Notas para el modelado

- `Heart Disease` en train es categórico (`Presence`/`Absence`), hay que codificarlo antes de entrenar (e.g. `Presence=1`, `Absence=0`).
- Variables como `Chest pain type`, `EKG results`, `Slope of ST` y `Thallium` son ordinales/categóricas y pueden beneficiarse de encoding.
- `ST depression` es la única variable continua no entera.
- No hay columna `Heart Disease` en el test set, solo `id` + las 13 features.
