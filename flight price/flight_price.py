"""
Domestic Flight Price Prediction Pipeline
Course Project - Machine Learning
-------------------------------------------------------------------------
This pipeline implements:
1. Exact deterministic airline recovery from flight IATA prefixes.
2. Hierarchical median imputation for flight durations.
3. High-order interaction feature engineering and Bayesian OOF Target Encoding.
4. Comprehensive 5-fold cross-validation benchmarking across standard linear,
   tree-based, and ensemble architectures.
5. Out-of-fold meta-stacking with positive-constrained Ridge regression.
6. Automated generation of submission files and metric summary tables.
"""

import time
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, root_mean_squared_error, mean_absolute_error
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor
)

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    from lightgbm import LGBMRegressor
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

try:
    from catboost import CatBoostRegressor
    HAS_CAT = True
except ImportError:
    HAS_CAT = False


def preprocess_and_engineer(train_df, test_df):
    """
    Executes domain-specific imputation and feature extraction.
    """
    full = pd.concat([train_df.drop(columns=['price']), test_df], ignore_index=True)

    # 1. Extract prefix and flight number
    full['flight_prefix'] = full['flight'].astype(str).str.split().str[0]
    full['flight_num'] = full['flight'].astype(str).str.split().str[1].astype(int)

    # 2. Deterministic prefix-to-carrier recovery
    prefix_to_airline = (
        full.dropna(subset=['airline'])
        .groupby('flight_prefix')['airline']
        .first()
        .to_dict()
    )
    full['airline_is_missing'] = full['airline'].isnull().astype(int)
    full['airline'] = full['airline'].fillna(full['flight_prefix'].map(prefix_to_airline))

    # 3. Hierarchical scheduled duration imputation
    full['duration_is_missing'] = full['duration'].isnull().astype(int)
    g1 = full.groupby(['flight', 'source_city', 'destination_city', 'stops', 'departure_time', 'arrival_time'])['duration'].transform('median')
    g2 = full.groupby(['flight', 'source_city', 'destination_city', 'stops'])['duration'].transform('median')
    g3 = full.groupby(['flight_prefix', 'source_city', 'destination_city', 'stops'])['duration'].transform('median')
    g4 = full.groupby(['source_city', 'destination_city', 'stops'])['duration'].transform('median')
    full['duration'] = (
        full['duration']
        .fillna(g1)
        .fillna(g2)
        .fillna(g3)
        .fillna(g4)
        .fillna(full['duration'].median())
    )

    # 4. Interaction Features
    full['route'] = full['source_city'] + '_' + full['destination_city']
    full['airline_class'] = full['airline'] + '_' + full['flight_class']
    full['prefix_class'] = full['flight_prefix'] + '_' + full['flight_class']
    full['route_class'] = full['route'] + '_' + full['flight_class']
    full['flight_class_combo'] = full['flight'] + '_' + full['flight_class']
    full['dep_arr'] = full['departure_time'] + '_' + full['arrival_time']
    full['route_stops_class'] = full['route'] + '_' + full['stops'] + '_' + full['flight_class']

    stops_map = {'zero': 0, 'one': 1, 'two_or_more': 2}
    full['stops_num'] = full['stops'].map(stops_map)
    full['is_business'] = (full['flight_class'] == 'Business').astype(int)

    full['flight_freq'] = full['flight'].map(full['flight'].value_counts())
    full['route_freq'] = full['route'].map(full['route'].value_counts())

    route_min_dur = full.groupby('route')['duration'].transform('min')
    full['duration_ratio_to_min'] = full['duration'] / (route_min_dur + 0.01)

    full['is_last_minute'] = (full['days_left'] <= 2).astype(int)
    full['is_urgent'] = (full['days_left'] <= 7).astype(int)
    full['log_days_left'] = np.log1p(full['days_left'])
    full['inv_days_left'] = 1.0 / (full['days_left'] + 1.0)
    full['log_duration'] = np.log1p(full['duration'])
    full['business_x_duration'] = full['is_business'] * full['duration']

    train_clean = full.iloc[:len(train_df)].copy()
    test_clean = full.iloc[len(train_df):].copy()
    return train_clean, test_clean


def main():
    t_start = time.time()
    print("[1/4] Loading datasets...")
    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')
    sub_df = pd.read_csv('sample_submission.csv')

    print("[2/4] Running feature engineering and out-of-fold target encoding...")
    X_train, X_test = preprocess_and_engineer(train_df, test_df)
    y_train = train_df['price'].values.astype(float)

    te_cols = [
        'route', 'airline_class', 'prefix_class', 'route_class',
        'dep_arr', 'route_stops_class', 'flight', 'flight_class_combo'
    ]
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    global_mean = y_train.mean()
    smooth = 10.0

    for col in te_cols:
        X_train[f'{col}_te'] = global_mean
        for tr_idx, val_idx in kf.split(X_train):
            tr_col = X_train.iloc[tr_idx][col]
            stats = pd.DataFrame({'col': tr_col, 'y': y_train[tr_idx]}).groupby('col')['y'].agg(['count', 'mean'])
            te_map = (stats['count'] * stats['mean'] + smooth * global_mean) / (stats['count'] + smooth)
            X_train.iloc[val_idx, X_train.columns.get_loc(f'{col}_te')] = (
                X_train.iloc[val_idx][col].map(te_map).fillna(global_mean).values
            )

        full_stats = pd.DataFrame({'col': X_train[col], 'y': y_train}).groupby('col')['y'].agg(['count', 'mean'])
        full_te_map = (full_stats['count'] * full_stats['mean'] + smooth * global_mean) / (full_stats['count'] + smooth)
        X_test[f'{col}_te'] = X_test[col].map(full_te_map).fillna(global_mean).values

    te_feature_names = [f'{col}_te' for col in te_cols]
    cat_cols_low = [
        'airline', 'flight_prefix', 'source_city', 'departure_time',
        'stops', 'arrival_time', 'destination_city', 'flight_class'
    ]
    ord_cols = ['flight', 'route', 'airline', 'departure_time', 'arrival_time', 'flight_class_combo']
    num_cols = [
        'duration', 'days_left', 'flight_num', 'stops_num', 'is_business',
        'flight_freq', 'route_freq', 'duration_ratio_to_min', 'is_last_minute',
        'is_urgent', 'log_days_left', 'inv_days_left', 'log_duration',
        'business_x_duration', 'airline_is_missing', 'duration_is_missing'
    ] + te_feature_names

    preproc_linear = ColumnTransformer([
        ('num', StandardScaler(), num_cols),
        ('cat_low', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols_low)
    ])

    preproc_tree = ColumnTransformer([
        ('num', 'passthrough', num_cols),
        ('cat_low', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols_low),
        ('cat_ord', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), ord_cols)
    ])

    models = {
        "Linear Regression (OLS)": Pipeline([('p', preproc_linear), ('m', LinearRegression())]),
        "Ridge Regression (L2)": Pipeline([('p', preproc_linear), ('m', Ridge(alpha=10.0, random_state=42))]),
        "Lasso Regression (L1)": Pipeline([('p', preproc_linear), ('m', Lasso(alpha=2.0, max_iter=1000, random_state=42))]),
        "Decision Tree Regressor": Pipeline([('p', preproc_tree), ('m', DecisionTreeRegressor(max_depth=14, min_samples_leaf=4, random_state=42))]),
        "HistGradientBoosting (Base)": Pipeline([('p', preproc_tree), ('m', HistGradientBoostingRegressor(max_iter=500, learning_rate=0.05, max_leaf_nodes=63, min_samples_leaf=10, l2_regularization=1.5, random_state=42))]),
        "HistGradientBoosting (Deep)": Pipeline([('p', preproc_tree), ('m', HistGradientBoostingRegressor(max_iter=700, learning_rate=0.035, max_leaf_nodes=127, min_samples_leaf=8, l2_regularization=2.5, random_state=99))]),
        "Random Forest Regressor": Pipeline([('p', preproc_tree), ('m', RandomForestRegressor(n_estimators=120, max_depth=22, max_features=0.5, random_state=42, n_jobs=-1))]),
        "Extra Trees Regressor": Pipeline([('p', preproc_tree), ('m', ExtraTreesRegressor(n_estimators=120, max_depth=24, max_features=0.5, random_state=42, n_jobs=-1))]),
        "Gradient Boosting Regressor": Pipeline([('p', preproc_tree), ('m', GradientBoostingRegressor(n_estimators=150, learning_rate=0.08, max_depth=6, subsample=0.8, max_features=0.6, random_state=42))])
    }

    if HAS_XGB:
        models["XGBoost Regressor"] = Pipeline([('p', preproc_tree), ('m', XGBRegressor(
            n_estimators=700, learning_rate=0.035, max_depth=8, subsample=0.8,
            colsample_bytree=0.8, reg_alpha=1.0, reg_lambda=2.0, tree_method='hist', random_state=42, n_jobs=-1
        ))])

    if HAS_LGB:
        models["LightGBM Regressor"] = Pipeline([('p', preproc_tree), ('m', LGBMRegressor(
            n_estimators=700, learning_rate=0.035, num_leaves=63,
            subsample=0.8, colsample_bytree=0.8, reg_alpha=1.0, reg_lambda=2.0, random_state=42, n_jobs=-1, verbose=-1
        ))])

    if HAS_CAT:
        models["CatBoost Regressor"] = Pipeline([('p', preproc_tree), ('m', CatBoostRegressor(
            iterations=800, learning_rate=0.045, depth=8, l2_leaf_reg=3.0, random_seed=42, thread_count=-1, verbose=0
        ))])

    print("[3/4] Benchmarking models with 5-fold cross-validation...")
    oof_preds = np.zeros((len(X_train), len(models)))
    test_preds = np.zeros((len(X_test), len(models)))
    model_names = list(models.keys())
    metrics_records = []

    print("=" * 86)
    print(f"{'Model Name':32s} | {'CV R2':10s} | {'CV RMSE':12s} | {'CV MAE':10s} | {'Time':6s}")
    print("=" * 86)

    for idx, name in enumerate(model_names):
        m_start = time.time()
        pipe = models[name]
        fold_oof = np.zeros(len(X_train))
        fold_test = np.zeros(len(X_test))

        for tr_idx, val_idx in kf.split(X_train, y_train):
            pipe.fit(X_train.iloc[tr_idx], y_train[tr_idx])
            fold_oof[val_idx] = pipe.predict(X_train.iloc[val_idx])
            fold_test += pipe.predict(X_test) / kf.n_splits

        oof_preds[:, idx] = fold_oof
        test_preds[:, idx] = fold_test

        r2 = r2_score(y_train, fold_oof)
        rmse = root_mean_squared_error(y_train, fold_oof)
        mae = mean_absolute_error(y_train, fold_oof)
        elapsed = time.time() - m_start

        print(f"{name:32s} | {r2:10.5f} | {rmse:12.2f} | {mae:10.2f} | {elapsed:5.1f}s")
        metrics_records.append({
            'Model': name,
            'CV_R2': round(r2, 5),
            'CV_RMSE': round(rmse, 2),
            'CV_MAE': round(mae, 2)
        })

        safe_name = name.replace(" ", "_").replace("(", "").replace(")", "")
        indiv_sub = sub_df.copy()
        indiv_sub['price'] = np.clip(fold_test, y_train.min(), None)
        indiv_sub.to_csv(f"sub_{safe_name}.csv", index=False)

    print("\n[4/4] Fitting Out-of-Fold Ridge Meta-Stacker...")
    strong_indices = [
        i for i, m in enumerate(model_names)
        if "Linear" not in m and "Ridge" not in m and "Lasso" not in m and "Decision Tree" not in m
    ]
    meta_oof = np.zeros(len(X_train))
    meta_test = np.zeros(len(X_test))

    meta_model = Ridge(alpha=100.0, positive=True)
    for tr_idx, val_idx in kf.split(oof_preds[:, strong_indices], y_train):
        meta_model.fit(oof_preds[tr_idx][:, strong_indices], y_train[tr_idx])
        meta_oof[val_idx] = meta_model.predict(oof_preds[val_idx][:, strong_indices])
        meta_test += meta_model.predict(test_preds[:, strong_indices]) / kf.n_splits

    min_price = y_train.min()
    meta_oof = np.clip(meta_oof, min_price, None)
    meta_test = np.clip(meta_test, min_price, None)

    ens_r2 = r2_score(y_train, meta_oof)
    ens_rmse = root_mean_squared_error(y_train, meta_oof)
    ens_mae = mean_absolute_error(y_train, meta_oof)
    total_time = time.time() - t_start

    print("=" * 86)
    print(f"{'Stacked Super-Ensemble (FINAL)':32s} | {ens_r2:10.5f} | {ens_rmse:12.2f} | {ens_mae:10.2f} | {total_time:5.1f}s")
    print("=" * 86)

    metrics_records.append({
        'Model': 'Stacked Super-Ensemble (FINAL)',
        'CV_R2': round(ens_r2, 5),
        'CV_RMSE': round(ens_rmse, 2),
        'CV_MAE': round(ens_mae, 2)
    })

    pd.DataFrame(metrics_records).to_csv('model_comparison_table.csv', index=False)
    sub_df['price'] = meta_test
    sub_df.to_csv('submission_flight_price.csv', index=False)
    print("\nPipeline finished. Outputs saved:")
    print("- submission_flight_price.csv (Final Competition Submission)")
    print("- model_comparison_table.csv (Report Metrics Table)")


if __name__ == '__main__':
    main()