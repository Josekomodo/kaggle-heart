# Project Structure

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

## Conventions
- All feature engineering, training, and inference code should reference data from `playground-series-s6e2/`
- Submission files must match `sample_submission.csv` format: two columns `id` and `Heart Disease`, one row per test ID
- Target encoding: `Presence=1`, `Absence=0` before training; output is a float probability [0, 1]
- Feature names contain spaces (e.g., `Chest pain type`, `Max HR`) — use bracket notation or rename on load
- `ST depression` is the only float feature; all others are integers
