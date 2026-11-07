"""
Iris Dataset Classification & Complete Evaluation Metrics
==========================================================
This script trains a classifier on the Scikit-Learn Iris dataset
and computes all essential multiclass classification evaluation metrics.
"""

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    log_loss,
    cohen_kappa_score,
    matthews_corrcoef,
    hamming_loss,
    zero_one_loss
)


def run_iris_classification():
    print("=" * 70)
    print("       IRIS DATASET CLASSIFICATION & EVALUATION METRICS")
    print("=" * 70)

    # 1. Load the Iris Dataset
    iris = load_iris()
    X = iris.data
    y = iris.target
    target_names = iris.target_names
    feature_names = iris.feature_names

    print(f"\n[1] Dataset Overview:")
    print(f"    - Total Samples: {X.shape[0]}")
    print(f"    - Number of Features: {X.shape[1]} ({', '.join(feature_names)})")
    print(f"    - Target Classes ({len(target_names)}): {list(target_names)}")
    print(f"    - Class Distribution: {[int(np.sum(y == i)) for i in range(len(target_names))]}")

    # 2. Train-Test Split (80% train, 20% test, stratified by class)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"\n[2] Data Splitting:")
    print(f"    - Training Set: {X_train.shape[0]} samples")
    print(f"    - Testing Set:  {X_test.shape[0]} samples (stratified)")

    # 3. Feature Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 4. Train Model (Random Forest Classifier)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)

    # Predictions
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)

    # 5. Calculate Evaluation Metrics
    print("\n" + "=" * 70)
    print("                 COMPREHENSIVE EVALUATION METRICS")
    print("=" * 70)

    # Accuracy Metrics
    acc = accuracy_score(y_test, y_pred)
    balanced_acc = balanced_accuracy_score(y_test, y_pred)
    zero_one = zero_one_loss(y_test, y_pred)
    hamming = hamming_loss(y_test, y_pred)

    print("\n--- 1. Accuracy & Error Metrics ---")
    print(f"  * Accuracy:                     {acc:.4f}  (Overall percentage of correct predictions)")
    print(f"  * Balanced Accuracy:            {balanced_acc:.4f}  (Average recall across each class)")
    print(f"  * Zero-One Loss (Error Rate):   {zero_one:.4f}  (Fraction of misclassified samples)")
    print(f"  * Hamming Loss:                 {hamming:.4f}  (Fraction of mispredicted labels)")

    # Precision, Recall, and F1 Metrics across Averaging Strategies
    # Macro: unweighted mean across classes
    # Weighted: mean weighted by class support
    # Micro: global total true positives / (TP + FP)
    prec_macro = precision_score(y_test, y_pred, average='macro')
    prec_weighted = precision_score(y_test, y_pred, average='weighted')
    rec_macro = recall_score(y_test, y_pred, average='macro')
    rec_weighted = recall_score(y_test, y_pred, average='weighted')
    f1_macro = f1_score(y_test, y_pred, average='macro')
    f1_weighted = f1_score(y_test, y_pred, average='weighted')

    print("\n--- 2. Precision, Recall & F1-Score (Averaged) ---")
    print(f"  * Precision (Macro Avg):        {prec_macro:.4f}  (Unweighted mean of precision)")
    print(f"  * Precision (Weighted Avg):     {prec_weighted:.4f}  (Precision weighted by class size)")
    print(f"  * Recall (Macro Avg):           {rec_macro:.4f}  (Unweighted mean of recall/sensitivity)")
    print(f"  * Recall (Weighted Avg):        {rec_weighted:.4f}  (Recall weighted by class size)")
    print(f"  * F1-Score (Macro Avg):         {f1_macro:.4f}  (Harmonic mean of precision & recall)")
    print(f"  * F1-Score (Weighted Avg):      {f1_weighted:.4f}  (Weighted harmonic mean)")

    # Probability-Based Metrics
    # Multiclass ROC-AUC using One-vs-Rest (OvR) and One-vs-One (OvO)
    roc_ovr = roc_auc_score(y_test, y_prob, multi_class='ovr', average='macro')
    roc_ovo = roc_auc_score(y_test, y_prob, multi_class='ovo', average='macro')
    loss = log_loss(y_test, y_prob)

    print("\n--- 3. Probability & Threshold Metrics ---")
    print(f"  * ROC-AUC (One-vs-Rest, Macro): {roc_ovr:.4f}  (Area under ROC curve comparing each class vs all)")
    print(f"  * ROC-AUC (One-vs-One, Macro):  {roc_ovo:.4f}  (Area under ROC comparing all unique class pairs)")
    print(f"  * Log Loss (Cross-Entropy):     {loss:.4f}  (Uncertainty/penalty of predicted probabilities)")

    # Agreement & Correlation Metrics
    kappa = cohen_kappa_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)

    print("\n--- 4. Statistical Correlation & Agreement ---")
    print(f"  * Cohen's Kappa Score:          {kappa:.4f}  (Agreement between actual & predicted, chance-corrected)")
    print(f"  * Matthews Correlation (MCC):   {mcc:.4f}  (Balanced quality metric from -1 to +1)")

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print("\n--- 5. Confusion Matrix ---")
    col_header = "       " + " ".join([f"{name:>12}" for name in target_names]) + "  (Predicted)"
    print(col_header)
    for i, row in enumerate(cm):
        row_str = f"{target_names[i]:>10}: " + " ".join([f"{val:>12}" for val in row])
        print(row_str)
    print("  (Actual)")

    # Per-Class Classification Report
    print("\n--- 6. Detailed Classification Report ---")
    print(classification_report(y_test, y_pred, target_names=target_names, digits=4))

    print("=" * 70)
    print("                        EXECUTION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    run_iris_classification()
