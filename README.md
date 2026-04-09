# Heart Disease Prediction

Kaggle Playground Series Season 6, Episode 2 — Binary classification competition predicting heart disease probability from 13 clinical features.

## Competition Overview

- **Task**: Binary classification (probabilistic output)
- **Evaluation Metric**: ROC AUC
- **Target**: `Heart Disease` (Presence/Absence)
- **Dataset**: Synthetic data generated from the Cleveland Heart Disease dataset
- **Submission**: Continuous probability [0, 1] of `Presence` per patient `id`

## Project Structure

```
.
├── playground-series-s6e2/     # Competition data
│   ├── train.csv               # Training data with target
│   ├── test.csv                # Test data (IDs 630000–633330)
│   ├── sample_submission.csv   # Submission format
│   └── README.md               # Competition description
├── src/
│   ├── __init__.py
│   └── eda.py                  # EDA pipeline
├── tests/
│   ├── conftest.py
│   └── test_eda.py             # Unit & property-based tests
├── output/
│   └── eda/                    # Generated EDA artifacts
│       ├── *.png               # Visualizations
│       └── conclusions.md      # EDA summary
├── .kiro/
│   ├── hooks/                  # Automation hooks
│   └── steering/               # Project guidelines
├── pyproject.toml              # Dependencies & tooling config
├── .pre-commit-config.yaml     # Pre-commit hooks
└── README.md
```

## Setup

### Prerequisites

- Python 3.9+
- `uv` (recommended) or `pip`

### Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

## Usage

### Run EDA Pipeline

```bash
python -m src.eda
```

Generates:
- 13 PNG visualizations in `eda_outputs/`
- `eda_outputs/conclusions.md` with findings and recommendations

### Run Tests

```bash
# All tests
pytest

# With coverage
pytest --cov=src --cov-report=html

# Property-based tests only
pytest -k property
```

## Development

### Code Quality

Pre-commit hooks run automatically on commit:
- **ruff**: Linting with auto-fix
- **ruff-format**: Code formatting
- **interrogate**: Docstring coverage (95% minimum)

Run manually:
```bash
pre-commit run --all-files
```

### Testing Strategy

- **Unit tests**: Validate individual function behavior
- **Property-based tests**: Use Hypothesis to verify correctness properties across random inputs
- **Requirements validation**: Each test maps to specific requirements

### Conventions

- **Target encoding**: `Presence=1`, `Absence=0`
- **Feature names**: Contain spaces — use bracket notation or rename on load
- **Numeric features**: `age`, `bp`, `cholesterol`, `max_hr`, `st_depression`
- **Categorical features**: 8 features including `sex`, `chest_pain_type`, etc.
- **Submission format**: Two columns `id` and `Heart Disease` (probability)

## Key Features

### EDA Pipeline

1. **Data Loading & Validation**: Shape, dtypes, nulls, duplicates
2. **Target Analysis**: Class distribution, imbalance detection
3. **Numeric Features**: Descriptive stats, zero anomaly detection, distributions, boxplots
4. **Categorical Features**: Chi-squared tests, presence rates, grouped bar charts
5. **Correlations**: Pearson matrix, Spearman feature-target rankings
6. **Train-Test Distribution**: KS tests, KDE overlays, drift detection
7. **Outlier Analysis**: IQR criterion, violin plots
8. **Feature Interactions**: Pairplot, top Spearman-correlated pairs
9. **Feature Importance**: Random Forest importances, 5-fold CV ROC AUC
10. **Conclusions**: Automated markdown report with recommendations

### Automated Recommendations

The pipeline detects and flags:
- Zero anomalies in `bp` and `cholesterol` (clinically impossible)
- Class imbalance (minority < 40%)
- Distribution shift between train/test (KS p-value < 0.05)
- IQR outliers per feature
- Top correlated features for modeling

## Contributing

1. Follow PEP 8 and project steering guidelines (`.kiro/steering/`)
2. Write NumPy-style docstrings for all public functions
3. Add tests for new functionality
4. Ensure pre-commit hooks pass before committing
5. Use conventional commit format: `type(scope): description`
