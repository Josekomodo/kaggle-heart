# Tech Stack

## Language & Environment
- Python 3.x
- Jupyter Notebooks or standalone scripts

## Core Libraries
- `pandas`, `numpy` — data manipulation
- `scikit-learn` — preprocessing, model training, evaluation (`roc_auc_score`)
- `xgboost`, `lightgbm`, or `catboost` — gradient boosting (typical top performers for tabular Kaggle competitions)
- `matplotlib`, `seaborn` — EDA and visualization

## Evaluation
```python
from sklearn.metrics import roc_auc_score
score = roc_auc_score(y_true, y_pred_proba)
```

## Common Commands
```bash
# Install dependencies
pip install pandas numpy scikit-learn xgboost lightgbm matplotlib seaborn

# Run a script
python train.py

# Launch notebook
jupyter notebook
```

## Data
- Input CSVs live in `playground-series-s6e2/`
- No external data sources required (self-contained competition dataset)
