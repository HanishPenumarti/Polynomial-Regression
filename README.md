# Polynomial Regression — AIT 511 Assignment 1

**Penumarti Hanish · BT2024190**

Polynomial regression for two prediction tasks: turbine Net Power Score (`var1`) and Thermal Anomaly Score (`var2`).

## Approach

- Generate polynomial terms, including lower-degree terms and interactions.
- Compare OLS using the pseudoinverse, Ridge, and Lasso.
- Select the degree and regularization strength using five-fold cross-validation.
- Scale polynomial columns within each training fold and group identical inputs together.
- Stop after three degrees without at least 1% improvement in the best validation MSE.
- Refit the selected model on all training rows and predict the test inputs.

## Run

Install dependencies:

```bash
pip install torch numpy pandas scikit-learn matplotlib
```

Place these files beside `code.py`:

```text
BT2024190_train_var1.csv
BT2024190_test_var1.csv
BT2024190_train_var2.csv
BT2024190_test_var2.csv
```

Then run:

```bash
python code.py
```

## Results

| Problem | Degree | Method | Code `alpha` | Validation MSE | Validation R² |
|---|---:|---|---:|---:|---:|
| var1 | 5 | Lasso | 0.01 | 0.34213935 | 0.96655128 |
| var2 | 10 | Ridge | 0.001 | 0.27560462 | 0.99326500 |

Scores are means across validation folds, not test scores; test targets are hidden.

## Outputs

The script prints selected configurations and validation scores, displays error plots, and saves:

- `predictions/BT2024190_pred_var1.csv`
- `predictions/BT2024190_pred_var2.csv`

Each CSV contains only the `y` column, preserving the original test-row order.
