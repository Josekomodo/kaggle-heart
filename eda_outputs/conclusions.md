# EDA Conclusions — Heart Disease Prediction

## 1. Overview

Dataset contains **630,000** training samples with 13 clinical features (5 numeric, 8 categorical) and a binary target (Heart Disease: Presence/Absence).

## 2. Class Balance

- Absence (0): 347,546 (55.2%)
- Presence (1): 282,454 (44.8%)

Minority class: **44.8%**

## 3. Numeric Features

### Descriptive Statistics

```
                     mean  median        std    min    max    p25    p75
age             54.136706    54.0   8.256301   29.0   77.0   48.0   60.0
bp             130.497433   130.0  14.975802   94.0  200.0  120.0  140.0
cholesterol    245.011814   243.0  33.681581  126.0  564.0  223.0  269.0
max_hr         152.816763   157.0  19.112927   71.0  202.0  142.0  166.0
st_depression    0.716028     0.1   0.948472    0.0    6.2    0.0    1.4
```

### Zero Anomaly Counts

- `bp`: **0** zero values (clinically anomalous)
- `cholesterol`: **0** zero values (clinically anomalous)

## 4. Categorical Features

### Chi-Squared Test p-values (vs Target)

| Feature | p-value | Significant? |
|---|---|---|
| sex | 0.0000e+00 | ✓ Yes |
| chest_pain_type | 0.0000e+00 | ✓ Yes |
| fbs_over_120 | 2.2808e-156 | ✓ Yes |
| ekg_results | 0.0000e+00 | ✓ Yes |
| exercise_angina | 0.0000e+00 | ✓ Yes |
| slope_of_st | 0.0000e+00 | ✓ Yes |
| num_vessels_fluro | 0.0000e+00 | ✓ Yes |
| thallium | 0.0000e+00 | ✓ Yes |

## 5. Correlations

**Top 3 features by |Spearman| correlation with target:** thallium, chest_pain_type, num_vessels_fluro

### Full Spearman Correlation Table (feature vs target)

| Feature | Spearman ρ |
|---|---|
| thallium | 0.6050 |
| chest_pain_type | 0.5089 |
| num_vessels_fluro | 0.4627 |
| exercise_angina | 0.4419 |
| max_hr | -0.4410 |
| st_depression | 0.4305 |
| slope_of_st | 0.4271 |
| sex | 0.3424 |
| ekg_results | 0.2190 |
| age | 0.2167 |
| cholesterol | 0.0912 |
| fbs_over_120 | 0.0336 |
| bp | 0.0008 |

## 6. Train-Test Shift

### KS Test Results

| Feature | KS Statistic | p-value | Problematic? |
|---|---|---|---|
| age | 0.0022 | 3.3931e-01 | No |
| bp | 0.0023 | 2.5686e-01 | No |
| cholesterol | 0.0014 | 8.5463e-01 | No |
| max_hr | 0.0017 | 6.1823e-01 | No |
| st_depression | 0.0026 | 1.6406e-01 | No |

No features with significant distribution shift detected (all p ≥ 0.05).

## 7. Outliers

### IQR Outlier Counts

| Feature | Count | Percentage |
|---|---|---|
| age | 1,048 | 0.17% |
| bp | 9,011 | 1.43% |
| cholesterol | 2,194 | 0.35% |
| max_hr | 14,246 | 2.26% |
| st_depression | 9,971 | 1.58% |

## 8. Feature Importance

### Random Forest — Top 5 Features

| Feature | Importance |
|---|---|
| thallium | 0.1796 |
| chest_pain_type | 0.1494 |
| max_hr | 0.1290 |
| cholesterol | 0.0870 |
| age | 0.0757 |

**CV ROC AUC (5-fold):** 0.9470 ± 0.0007

## 9. Recommendations

- **Cap or robustly scale outliers** (e.g., RobustScaler or winsorization at 1st/99th percentile).
- **Encode categorical features** (ordinal or target encoding for tree models; one-hot for linear models).
- **Scale numeric features** (StandardScaler or RobustScaler) for distance-based and linear models.
- **Use ROC AUC** as the primary evaluation metric throughout model development.
