# Domestic Flight Price Prediction Challenge

A machine learning solution for predicting domestic airline ticket prices, combining deterministic categorical imputation, domain feature engineering, Bayesian Out-of-Fold Target Encoding, and an Out-of-Fold Ridge-Stacked Super-Ensemble.

---

## Project Structure

.
├── flight_price_pipeline.py    # Main training, cross-validation & prediction pipeline
├── generate_eda_figures.py     # Reproducible script for report charts & EDA figures
├── eda_and_model_results.png   # 4-panel EDA and model benchmark visualization
├── model_comparison_table.csv  # 5-fold cross-validation metrics across all models
├── requirements.txt            # Python dependencies
├── .gitignore                  # Git ignore rules
└── README.md                   # Repository documentation

---

## Key Methodological Highlights

1. **Deterministic Categorical Imputation:**
   * Missing `airline` values (~20% of dataset) were mapped with 100% precision using a 1-to-1 deterministic relationship with the 2-letter IATA `flight_prefix`.
   * Missing `duration` values (~20% of dataset) were imputed using a 4-tier hierarchical scheduled block lookup (`flight + route + stops` -> `airline + route + stops` -> `route + stops`), resolving 97.75% of missing durations to their scheduled value.

2. **Domain-Specific Feature Engineering:**
   * Multiplicative interactions: `route` (`source_city` + `destination_city`), `airline_class`, `flight_class_combo`, and `route_stops_class`.
   * Out-of-Fold Smoothed Target Encoding ($m = 10.0$) across high-cardinality nominals to capture granular route/flight price tiers without data leakage.
   * Booking window non-linearities: `inv_days_left`, `log_days_left`, and `duration_ratio_to_min`.

3. **Benchmarking & Meta-Stacking:**
   * Evaluated across classical linear algorithms (OLS, Ridge, Lasso), Decision Trees, and Ensemble Methods (Random Forest, Extra Trees, Gradient Boosting, HistGradientBoosting).
   * Combined out-of-fold predictions into a non-negative Ridge Meta-Estimator, achieving the top $R^2$ of **0.92800** and RMSE of **10,498.96**.

---

## 5-Fold Cross-Validation Benchmark Results

| Model | CV $R^2$ Score | CV RMSE | CV MAE |
| :--- | :---: | :---: | :---: |
| **Linear Regression (OLS)** | 0.90409 | 12,117.33 | 5,529.13 |
| **Ridge Regression ($L_2$)** | 0.90274 | 12,202.38 | 5,552.89 |
| **Lasso Regression ($L_1$)** | 0.90393 | 12,127.13 | 5,538.87 |
| **Decision Tree Regressor** | 0.89442 | 12,713.68 | 4,187.25 |
| **HistGradientBoosting (Deep)** | 0.91862 | 11,161.90 | 3,969.97 |
| **HistGradientBoosting (Base)** | 0.92097 | 10,999.39 | 4,003.30 |
| **Gradient Boosting Regressor** | 0.92371 | 10,806.97 | 3,870.94 |
| **Extra Trees Regressor** | 0.92373 | 10,805.37 | **3,366.95** |
| **Random Forest Regressor** | 0.92666 | 10,596.22 | 3,528.02 |
| **Stacked Super-Ensemble (FINAL)** | **0.92800** | **10,498.96** | 3,472.43 |

---

## Quickstart

### 1. Installation
Clone the repository and install the dependencies:
```bash
git clone https://github.com/<your-username>/domestic-flight-price-prediction.git
cd domestic-flight-price-prediction
pip install -r requirements.txt
```


### 2. Execution
Place train.csv, test.csv, and sample_submission.csv in the root folder, then run:
python flight_price_pipeline.py

This will run the 5-fold cross-validation pipeline, generate submission_flight_price.csv, and export model_comparison_table.csv.

To regenerate the EDA visualization figures:

python generate_eda_figures.py
