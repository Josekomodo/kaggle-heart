# Project Structure

```
playground-series-s6e2/
├── train.csv              # Training data with target column `Heart Disease`
├── test.csv               # Test data (no target); IDs from 630000–633330
├── sample_submission.csv  # Expected submission format: id, Heart Disease (probability)
└── README.md              # Competition description, features, and modeling notes
```

## Conventions
- All feature engineering, training, and inference code should reference data from `playground-series-s6e2/`
- Submission files must match `sample_submission.csv` format: two columns `id` and `Heart Disease`, one row per test ID
- Target encoding: `Presence=1`, `Absence=0` before training; output is a float probability [0, 1]
- Feature names contain spaces (e.g., `Chest pain type`, `Max HR`) — use bracket notation or rename on load
- `ST depression` is the only float feature; all others are integers
