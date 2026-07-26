"""
Train the final fraud detection model and save everything the dashboard needs.

Outputs (into artifacts/):
    fraud_model.joblib      trained XGBoost model
    model_metadata.json     features, threshold, metrics, cost assumptions
    eda_summary.json        precomputed aggregates so the app runs standalone
"""
import json
import os
import time
import warnings

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import (average_precision_score, roc_auc_score,
                             precision_recall_curve, confusion_matrix)

warnings.filterwarnings('ignore')

DATA_PATH = '/home/claude/paysim/data/raw/paysim.csv'
ARTIFACT_DIR = '/home/claude/paysim/artifacts'
RANDOM_STATE = 42

os.makedirs(ARTIFACT_DIR, exist_ok=True)


def find_low_activity_positions(df, quantile=0.33):
    """Which cycle positions are quietest, judged purely on observed volume.

    The data card tells us one step is one hour and the simulation covers 30 days.
    It does not tell us what clock time step 1 corresponds to, so we must not label
    these positions with real hours. We only claim what the data supports: a
    repeating 24 step cycle in which some positions carry far less traffic.
    """
    volume = df.groupby(df['step'] % 24).size()
    return sorted(volume.index[volume <= volume.quantile(quantile)].tolist())


def build_features(df, low_activity_positions):
    """Create every model feature. Kept identical to the notebook."""
    df = df.copy()
    # Position within the repeating 24 step cycle. Not a clock hour: we have no
    # information about when the simulation began.
    df['cycle_position'] = df['step'] % 24
    df['is_low_activity'] = df['cycle_position'].isin(low_activity_positions).astype(int)
    df['log_amount'] = np.log1p(df['amount'])
    df['log_old_org'] = np.log1p(df['oldbalanceOrg'].clip(lower=0))
    df['log_old_dest'] = np.log1p(df['oldbalanceDest'].clip(lower=0))
    return df


print('Loading data')
raw = pd.read_csv(DATA_PATH)

LOW_ACTIVITY = find_low_activity_positions(raw)
print(f'Low activity cycle positions, derived from volume: {LOW_ACTIVITY}')
df = build_features(raw, LOW_ACTIVITY)

# One hot encoding, dropping the first category to avoid the dummy variable trap
df = pd.get_dummies(df, columns=['type'], prefix='type', dtype=int, drop_first=True)
type_cols = sorted(c for c in df.columns if c.startswith('type_'))

FEATURES = (['log_amount', 'cycle_position', 'is_low_activity',
             'log_old_org', 'log_old_dest'] + type_cols)
print(f'Features: {FEATURES}')

# Three way split, stratified so the fraud rate stays constant
train_full, test = train_test_split(df, test_size=0.30, random_state=RANDOM_STATE,
                                    stratify=df['isFraud'])
train, val = train_test_split(train_full, test_size=0.20, random_state=RANDOM_STATE,
                              stratify=train_full['isFraud'])

X_train, y_train = train[FEATURES].astype(float), train['isFraud'].values
X_val, y_val = val[FEATURES].astype(float), val['isFraud'].values
X_test, y_test = test[FEATURES].astype(float), test['isFraud'].values

scale_pos = (y_train == 0).sum() / (y_train == 1).sum()
print(f'scale_pos_weight = {scale_pos:.0f}')

print('Training')
started = time.time()
model = xgb.XGBClassifier(
    n_estimators=400, max_depth=5, learning_rate=0.1, min_child_weight=5,
    scale_pos_weight=scale_pos, eval_metric='aucpr', tree_method='hist',
    n_jobs=-1, random_state=RANDOM_STATE)
model.fit(X_train, y_train)
print(f'Trained in {time.time() - started:.1f}s')

val_proba = model.predict_proba(X_val)[:, 1]
test_proba = model.predict_proba(X_test)[:, 1]


def best_f1_threshold(y_true, proba):
    precision, recall, thresholds = precision_recall_curve(y_true, proba)
    f1 = 2 * precision * recall / (precision + recall + 1e-12)
    return float(thresholds[np.nanargmax(f1[:-1])])


def best_cost_threshold(y_true, proba, cost_fn, cost_fp):
    """Threshold minimising total business cost."""
    candidates = np.unique(np.quantile(proba, np.linspace(0.90, 0.99999, 400)))
    best_t, best_cost = 0.5, np.inf
    for t in candidates:
        pred = (proba >= t).astype(int)
        fn = int(((pred == 0) & (y_true == 1)).sum())
        fp = int(((pred == 1) & (y_true == 0)).sum())
        total = fn * cost_fn + fp * cost_fp
        if total < best_cost:
            best_cost, best_t = total, float(t)
    return best_t


# Business cost assumptions, stated openly so they can be challenged
COST_MISSED_FRAUD = 5000.0    # average rand value lost when a fraud goes through
COST_FALSE_ALARM = 50.0       # analyst time to review one alert

f1_threshold = best_f1_threshold(y_val, val_proba)
cost_threshold = best_cost_threshold(y_val, val_proba, COST_MISSED_FRAUD, COST_FALSE_ALARM)


def evaluate(y_true, proba, threshold):
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    return {
        'threshold': round(float(threshold), 6),
        'precision': round(float(precision), 4),
        'recall': round(float(recall), 4),
        'f1': round(float(2 * precision * recall / max(precision + recall, 1e-12)), 4),
        'true_positives': int(tp), 'false_positives': int(fp),
        'false_negatives': int(fn), 'true_negatives': int(tn),
        'alerts': int(tp + fp),
        'total_cost': float(fn * COST_MISSED_FRAUD + fp * COST_FALSE_ALARM),
    }


metrics_f1 = evaluate(y_test, test_proba, f1_threshold)
metrics_cost = evaluate(y_test, test_proba, cost_threshold)

# A no model baseline: approve everything, pay for every fraud
baseline_cost = float(y_test.sum() * COST_MISSED_FRAUD)

# Precision recall curve, thinned for plotting
precision, recall, thresholds = precision_recall_curve(y_test, test_proba)
keep = np.linspace(0, len(precision) - 1, 300).astype(int)

# Feature importance
importance = dict(zip(FEATURES, model.feature_importances_.astype(float).round(5).tolist()))

metadata = {
    'features': FEATURES,
    'low_activity_positions': LOW_ACTIVITY,
    'type_columns': type_cols,
    'model_type': 'XGBClassifier',
    'trained_rows': int(len(X_train)),
    'test_rows': int(len(X_test)),
    'test_fraud_count': int(y_test.sum()),
    'test_fraud_rate': float(y_test.mean()),
    'pr_auc': float(average_precision_score(y_test, test_proba)),
    'roc_auc': float(roc_auc_score(y_test, test_proba)),
    'thresholds': {'f1_optimal': metrics_f1, 'cost_optimal': metrics_cost},
    'costs': {'missed_fraud': COST_MISSED_FRAUD, 'false_alarm': COST_FALSE_ALARM,
              'baseline_no_model': baseline_cost},
    'importance': importance,
    'pr_curve': {'precision': precision[keep].round(4).tolist(),
                 'recall': recall[keep].round(4).tolist()},
    'accuracy_trap': {
        'always_normal_accuracy': float((y_test == 0).mean()),
        'model_accuracy': float(((test_proba >= cost_threshold).astype(int) == y_test).mean()),
    },
}

# Model comparison table from the notebook, for the dashboard
metadata['model_comparison'] = [
    {'model': 'Logistic Regression (baseline)', 'basic': 0.055, 'balances': 0.091, 'errors': 0.466},
    {'model': 'Decision Tree', 'basic': 0.236, 'balances': 0.198, 'errors': 0.272},
    {'model': 'Random Forest', 'basic': 0.204, 'balances': 0.716, 'errors': 0.845},
    {'model': 'XGBoost', 'basic': 0.283, 'balances': 0.836, 'errors': 0.990},
]

# ---------------------------------------------------------------- EDA summary
fraud_by_type = (raw.groupby('type')['isFraud']
                 .agg(transactions='size', fraud='sum', rate='mean')
                 .reset_index())

cycle = (raw.assign(cycle_position=raw['step'] % 24).groupby('cycle_position')['isFraud']
         .agg(transactions='size', fraud='sum', rate='mean').reset_index())

amt_bins = [0, 1e3, 1e4, 5e4, 1e5, 5e5, 1e6, 1e7, 1e9]
amt_labels = ['0 to 1k', '1k to 10k', '10k to 50k', '50k to 100k',
              '100k to 500k', '500k to 1m', '1m to 10m', 'over 10m']
amt = raw.assign(band=pd.cut(raw['amount'], bins=amt_bins, labels=amt_labels))
amount_bands = (amt.groupby('band', observed=True)['isFraud']
                .agg(transactions='size', fraud='sum', rate='mean').reset_index())
amount_bands['band'] = amount_bands['band'].astype(str)

log_amt = np.log1p(raw['amount'])
hist_normal, edges = np.histogram(log_amt[raw.isFraud == 0], bins=40)
hist_fraud, _ = np.histogram(log_amt[raw.isFraud == 1], bins=edges)

eda = {
    'total_rows': int(len(raw)),
    'fraud_count': int(raw['isFraud'].sum()),
    'fraud_rate': float(raw['isFraud'].mean()),
    'normal_count': int((raw['isFraud'] == 0).sum()),
    'fraud_by_type': fraud_by_type.to_dict('records'),
    'cycle': cycle.to_dict('records'),
    'amount_bands': amount_bands.to_dict('records'),
    'amount_hist': {'edges': edges.round(3).tolist(),
                    'normal': hist_normal.tolist(), 'fraud': hist_fraud.tolist()},
    'amount_stats': {
        'normal_median': float(raw[raw.isFraud == 0]['amount'].median()),
        'fraud_median': float(raw[raw.isFraud == 1]['amount'].median()),
        'normal_mean': float(raw[raw.isFraud == 0]['amount'].mean()),
        'fraud_mean': float(raw[raw.isFraud == 1]['amount'].mean()),
    },
    'low_activity_positions': LOW_ACTIVITY,
    'activity_split': {
        'low_rate': float(raw[(raw['step'] % 24).isin(LOW_ACTIVITY)]['isFraud'].mean()),
        'busy_rate': float(raw[~(raw['step'] % 24).isin(LOW_ACTIVITY)]['isFraud'].mean()),
        'low_volume_share': float((raw['step'] % 24).isin(LOW_ACTIVITY).mean()),
    },
}

joblib.dump(model, f'{ARTIFACT_DIR}/fraud_model.joblib')
with open(f'{ARTIFACT_DIR}/model_metadata.json', 'w') as fh:
    json.dump(metadata, fh, indent=2)
with open(f'{ARTIFACT_DIR}/eda_summary.json', 'w') as fh:
    json.dump(eda, fh, indent=2)

print('\nSaved artifacts:')
for name in sorted(os.listdir(ARTIFACT_DIR)):
    size = os.path.getsize(f'{ARTIFACT_DIR}/{name}') / 1024
    print(f'  {name:24s} {size:8.1f} KB')

print(f'\nPR-AUC {metadata["pr_auc"]:.4f}   ROC-AUC {metadata["roc_auc"]:.4f}')
print(f'F1 threshold   {f1_threshold:.5f} -> precision {metrics_f1["precision"]}, '
      f'recall {metrics_f1["recall"]}, cost {metrics_f1["total_cost"]:,.0f}')
print(f'Cost threshold {cost_threshold:.5f} -> precision {metrics_cost["precision"]}, '
      f'recall {metrics_cost["recall"]}, cost {metrics_cost["total_cost"]:,.0f}')
print(f'No model at all -> cost {baseline_cost:,.0f}')
