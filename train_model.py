"""
Credit Card Fraud Detection - Model Training Script
====================================================
Author: Yash Sajlendra Pandey
Description: Training script for XGBoost fraud detection model with SMOTE balancing
Dataset: Credit Card Fraud Detection Dataset
"""

import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    classification_report
)
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import warnings

warnings.filterwarnings('ignore')

print("=" * 80)
print("CREDIT CARD FRAUD DETECTION - MODEL TRAINING")
print("=" * 80)

# ==============================
# 1. LOAD DATA
# ==============================
print("\n📥 Loading dataset...")
try:
    df = pd.read_csv("dataset.csv")  # Change path as needed
    print(f"✅ Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
except FileNotFoundError:
    print("❌ Error: 'dataset.csv' not found. Please provide the correct path.")
    exit(1)

# ==============================
# 2. EXPLORATORY DATA ANALYSIS
# ==============================
print("\n📊 Data Overview:")
print(f"   Missing values: {df.isnull().sum().sum()}")
print(f"   Data types:\n{df.dtypes.value_counts()}")
print(f"\n   Class distribution:")
print(df["Class"].value_counts())
print(f"   Fraud rate: {df['Class'].mean() * 100:.2f}%")

# ==============================
# 3. PREPARE FEATURES & TARGET
# ==============================
print("\n🔧 Preparing features and target...")
X = df.drop("Class", axis=1)
y = df["Class"]

print(f"   Features: {X.shape}")
print(f"   Target: {y.shape}")

# ==============================
# 4. TRAIN-TEST SPLIT
# ==============================
print("\n📈 Splitting data (80-20)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f"   Training set: {X_train.shape}")
print(f"   Test set: {X_test.shape}")
print(f"   Training fraud rate: {y_train.mean() * 100:.2f}%")
print(f"   Test fraud rate: {y_test.mean() * 100:.2f}%")

# ==============================
# 5. HANDLE CLASS IMBALANCE WITH SMOTE
# ==============================
print("\n⚖️  Applying SMOTE for class balancing...")
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

print(f"   After SMOTE:")
print(f"   Training set size: {X_train_balanced.shape}")
print(f"   Class distribution:\n{pd.Series(y_train_balanced).value_counts()}")
print(f"   Fraud rate: {pd.Series(y_train_balanced).mean() * 100:.2f}%")

# ==============================
# 6. FEATURE SCALING
# ==============================
print("\n📐 Scaling features with StandardScaler...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_balanced)
X_test_scaled = scaler.transform(X_test)

print(f"   Mean (training): {X_train_scaled.mean():.4f}")
print(f"   Std (training): {X_train_scaled.std():.4f}")

# ==============================
# 7. TRAIN XGBOOST MODEL
# ==============================
print("\n🤖 Training XGBoost Model...")
print("   Parameters:")
print("   - n_estimators: 100")
print("   - max_depth: 6")
print("   - learning_rate: 0.1")
print("   - random_state: 42")

model = XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    random_state=42,
    verbosity=0,
    use_label_encoder=False,
    eval_metric='logloss'
)

model.fit(X_train_scaled, y_train_balanced)
print("✅ Model training completed!")

# ==============================
# 8. MAKE PREDICTIONS
# ==============================
print("\n🔮 Making predictions on test set...")
y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
y_pred = (y_pred_proba > 0.7).astype(int)

print(f"   Predicted frauds: {y_pred.sum()}")
print(f"   Predicted legitimate: {(1 - y_pred).sum()}")

# ==============================
# 9. MODEL EVALUATION
# ==============================
print("\n📋 Model Evaluation Metrics:")
print("=" * 80)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)

print(f"   Accuracy:  {accuracy:.4f}")
print(f"   Precision: {precision:.4f}")
print(f"   Recall:    {recall:.4f}")
print(f"   F1-Score:  {f1:.4f}")
print(f"   ROC-AUC:   {roc_auc:.4f}")

print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred, target_names=['Legitimate', 'Fraud']))

# ==============================
# 10. CONFUSION MATRIX
# ==============================
print("\n🎯 Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(f"   True Negatives:  {cm[0][0]}")
print(f"   False Positives: {cm[0][1]}")
print(f"   False Negatives: {cm[1][0]}")
print(f"   True Positives:  {cm[1][1]}")

# ==============================
# 11. VISUALIZATIONS
# ==============================
print("\n📈 Generating visualizations...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.patch.set_facecolor('#0F1419')

# Confusion Matrix
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0], cbar=False)
axes[0, 0].set_title('Confusion Matrix', fontsize=12, fontweight='bold', color='white')
axes[0, 0].set_ylabel('True Label', color='white')
axes[0, 0].set_xlabel('Predicted Label', color='white')
axes[0, 0].tick_params(colors='white')

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
axes[0, 1].plot(fpr, tpr, color='#0066FF', linewidth=2, label=f'AUC = {roc_auc:.4f}')
axes[0, 1].plot([0, 1], [0, 1], 'k--', linewidth=1)
axes[0, 1].set_xlabel('False Positive Rate', color='white')
axes[0, 1].set_ylabel('True Positive Rate', color='white')
axes[0, 1].set_title('ROC Curve', fontsize=12, fontweight='bold', color='white')
axes[0, 1].legend(loc='lower right', facecolor='#1a1f2e', edgecolor='white')
axes[0, 1].tick_params(colors='white')
axes[0, 1].set_facecolor('#1a1f2e')

# Feature Importance (Top 15)
importance_df = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False).head(15)

axes[1, 0].barh(importance_df['feature'], importance_df['importance'], color='#0066FF')
axes[1, 0].set_xlabel('Importance Score', color='white')
axes[1, 0].set_title('Top 15 Feature Importances', fontsize=12, fontweight='bold', color='white')
axes[1, 0].tick_params(colors='white')
axes[1, 0].set_facecolor('#1a1f2e')
for spine in axes[1, 0].spines.values():
    spine.set_edgecolor('white')

# Probability Distribution
axes[1, 1].hist(y_pred_proba[y_test == 0], bins=30, alpha=0.7, label='Legitimate', color='#00D084')
axes[1, 1].hist(y_pred_proba[y_test == 1], bins=30, alpha=0.7, label='Fraud', color='#FF3366')
axes[1, 1].axvline(0.7, color='orange', linestyle='--', linewidth=2, label='Threshold')
axes[1, 1].set_xlabel('Fraud Probability', color='white')
axes[1, 1].set_ylabel('Count', color='white')
axes[1, 1].set_title('Fraud Probability Distribution', fontsize=12, fontweight='bold', color='white')
axes[1, 1].legend(facecolor='#1a1f2e', edgecolor='white')
axes[1, 1].tick_params(colors='white')
axes[1, 1].set_facecolor('#1a1f2e')
for spine in axes[1, 1].spines.values():
    spine.set_edgecolor('white')

plt.tight_layout()
plt.savefig('model_evaluation.png', dpi=300, bbox_inches='tight', facecolor='#0F1419')
print("   ✅ Saved: model_evaluation.png")

# ==============================
# 12. SAVE MODEL & SCALER
# ==============================
print("\n💾 Saving model and scaler...")

pickle.dump(model, open("xgb_model.pkl", "wb"))
pickle.dump(scaler, open("xgb_scaler.pkl", "wb"))

print("   ✅ xgb_model.pkl")
print("   ✅ xgb_scaler.pkl")

# ==============================
# 13. SAVE FEATURE COLUMNS
# ==============================
print("\n📝 Saving feature columns...")
feature_list = list(X.columns)
pickle.dump(feature_list, open("feature_columns.pkl", "wb"))
print("   ✅ feature_columns.pkl")

# ==============================
# 14. SUMMARY REPORT
# ==============================
print("\n" + "=" * 80)
print("MODEL TRAINING SUMMARY")
print("=" * 80)

summary = f"""
✅ Training Completed Successfully!

📊 Dataset Statistics:
   - Total samples: {len(df):,}
   - Features: {X.shape[1]}
   - Fraud cases: {y.sum():,} ({y.mean()*100:.2f}%)
   - Legitimate cases: {(1-y).sum():,} ({(1-y).mean()*100:.2f}%)

🔧 Data Processing:
   - Train-Test Split: 80-20
   - Class Balancing: SMOTE
   - Feature Scaling: StandardScaler

🤖 Model Configuration:
   - Algorithm: XGBoost
   - Estimators: 100
   - Max Depth: 6
   - Learning Rate: 0.1
   - Random State: 42

📈 Model Performance:
   - Accuracy:  {accuracy:.4f}
   - Precision: {precision:.4f}
   - Recall:    {recall:.4f}
   - F1-Score:  {f1:.4f}
   - ROC-AUC:   {roc_auc:.4f}

💾 Saved Files:
   - xgb_model.pkl (Model weights)
   - xgb_scaler.pkl (Feature scaler)
   - feature_columns.pkl (Feature list)
   - model_evaluation.png (Visualizations)

🚀 Ready for Deployment!
"""

print(summary)

# ==============================
# 15. DEPLOYMENT CHECKLIST
# ==============================
print("\n" + "=" * 80)
print("DEPLOYMENT CHECKLIST")
print("=" * 80)
print("""
✅ Model training completed
✅ Model saved as pickle file
✅ Scaler saved as pickle file
✅ Feature columns saved
✅ Evaluation metrics calculated
✅ Visualizations generated

📋 Next Steps:
1. Copy xgb_model.pkl to Streamlit app directory
2. Copy xgb_scaler.pkl to Streamlit app directory
3. Run: streamlit run app.py
4. Upload CSV files for fraud detection
5. Monitor and analyze predictions

📧 For questions or improvements: sajlendrapandey2022@gmail.com
""")

print("=" * 80)
