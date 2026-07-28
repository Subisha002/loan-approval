import os
import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import xgboost as xgb

def run_training_pipeline():
    print("1. Loading Dataset...")
    csv_path = 'loan_approval_dataset.csv'
    if not os.path.exists(csv_path):
        url = 'https://raw.githubusercontent.com/sarahrafiqshaikh/Loan-Approval-Prediction-Analysis/main/loan_approval_dataset.csv'
        print(f"Downloading dataset from {url}...")
        df = pd.read_csv(url)
        df.to_csv(csv_path, index=False)
    else:
        df = pd.read_csv(csv_path)

    # Preprocessing
    df.columns = df.columns.str.strip()
    if 'loan_id' in df.columns:
        df = df.drop('loan_id', axis=1)

    # Encode categorical columns
    df['education'] = df['education'].apply(lambda x: 0 if str(x).strip() == 'Graduate' else 1)
    df['self_employed'] = df['self_employed'].apply(lambda x: 0 if str(x).strip() == 'No' else 1)
    df['loan_status'] = df['loan_status'].apply(lambda x: 1 if str(x).strip() == 'Approved' else 0)

    X = df.drop('loan_status', axis=1)
    y = df['loan_status']
    feature_columns = list(X.columns)

    print(f"Features ({len(feature_columns)}): {feature_columns}")

    # Train Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Baseline Models Evaluation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    baseline_models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=8),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    }

    cv_results = {}
    fitted_baselines = {}
    for name, model in baseline_models.items():
        scores = cross_val_score(model, X_train_scaled, y_train, cv=cv, scoring='roc_auc')
        cv_results[name] = np.mean(scores)
        model.fit(X_train_scaled, y_train)
        fitted_baselines[name] = model

    # XGBoost Model
    pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    best_xgb = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        random_state=42,
        scale_pos_weight=pos_weight,
        n_jobs=-1,
        eval_metric='logloss'
    )
    best_xgb.fit(X_train_scaled, y_train)

    # Predictions & Test Metrics
    y_pred = best_xgb.predict(X_test_scaled)
    y_prob = best_xgb.predict_proba(X_test_scaled)[:, 1]

    test_metrics = {
        'ROC-AUC': float(roc_auc_score(y_test, y_prob)),
        'F1-Score': float(f1_score(y_test, y_pred)),
        'Accuracy': float(accuracy_score(y_test, y_pred)),
        'Precision': float(precision_score(y_test, y_pred)),
        'Recall': float(recall_score(y_test, y_pred))
    }

    # Summary table
    results_df = pd.DataFrame({
        'Model': list(cv_results.keys()) + ['XGBoost (Tuned)'],
        'ROC-AUC CV': [round(v, 3) for v in cv_results.values()] + [round(test_metrics['ROC-AUC'], 3)],
        'Test F1': [
            round(f1_score(y_test, fitted_baselines['Logistic Regression'].predict(X_test_scaled)), 3),
            round(f1_score(y_test, fitted_baselines['Decision Tree'].predict(X_test_scaled)), 3),
            round(f1_score(y_test, fitted_baselines['Random Forest'].predict(X_test_scaled)), 3),
            round(test_metrics['F1-Score'], 3)
        ]
    })

    print("\n--- MODEL PERFORMANCE METRICS ---")
    print(results_df)

    # Export PKL files
    joblib.dump(best_xgb, 'loan_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    joblib.dump(feature_columns, 'features.pkl')
    joblib.dump(test_metrics, 'metrics.pkl')
    joblib.dump(results_df, 'results_df.pkl')

    print("\n  All model files dumped successfully:")
    print("- loan_model.pkl")
    print("- scaler.pkl")
    print("- features.pkl")
    print("- metrics.pkl")
    print("- results_df.pkl")

if __name__ == '__main__':
    run_training_pipeline()
