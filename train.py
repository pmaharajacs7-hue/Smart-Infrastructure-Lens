import pandas as pd
import numpy as np
import joblib, os, warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except:
    HAS_XGB = False

# Load
df = pd.read_csv("data/pipeline_dataset.csv")
FEATURES = [
    "Station_pressure","J1_pressure","J2_pressure","J3_pressure","Home_pressure",
    "Flow_Sta_J1","Flow_J1_J2","Flow_J2_J3","Flow_J3_Home",
    "Pressure_drop_J1","Pressure_drop_J2","Pressure_drop_J3",
    "Flow_loss_J1J2","Flow_loss_J2J3"
]
X = df[FEATURES]
y = df["label"]

print("="*55)
print("  WATER PIPELINE — ML MODEL TRAINING")
print("="*55)
print(f"\nDataset: {len(X)} rows, {len(FEATURES)} features")
print(f"Normal: {sum(y==0)} | Moderate: {sum(y==1)} | Critical: {sum(y==2)}")

# Split & Scale
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# Model 1: Random Forest
print("\n🌲 Training Random Forest...")
rf = RandomForestClassifier(n_estimators=200, max_depth=15,
     class_weight="balanced", random_state=42, n_jobs=-1)
rf.fit(X_train_sc, y_train)
rf_acc = accuracy_score(y_test, rf.predict(X_test_sc))
print(f"   Accuracy: {rf_acc*100:.2f}%")

# Model 2: XGBoost or GradientBoosting
if HAS_XGB:
    print("\n⚡ Training XGBoost...")
    boost = XGBClassifier(n_estimators=200, max_depth=6,
            learning_rate=0.1, use_label_encoder=False,
            eval_metric="mlogloss", random_state=42)
    bname = "XGBoost"
else:
    print("\n⚡ Training GradientBoosting...")
    boost = GradientBoostingClassifier(n_estimators=200,
            max_depth=6, learning_rate=0.1, random_state=42)
    bname = "GradientBoosting"

boost.fit(X_train_sc, y_train)
boost_acc = accuracy_score(y_test, boost.predict(X_test_sc))
print(f"   Accuracy: {boost_acc*100:.2f}%")

# Model 3: Voting Ensemble
print("\n🗳️  Training Ensemble...")
ensemble = VotingClassifier([("rf", rf), ("boost", boost)], voting="soft")
ensemble.fit(X_train_sc, y_train)
ens_pred = ensemble.predict(X_test_sc)
ens_acc  = accuracy_score(y_test, ens_pred)
print(f"   Accuracy: {ens_acc*100:.2f}%")

# Pick Best
best_acc, best_model, best_pred, best_name = max(
    [(rf_acc, rf, rf.predict(X_test_sc), "Random Forest"),
     (boost_acc, boost, boost.predict(X_test_sc), bname),
     (ens_acc, ensemble, ens_pred, "Ensemble")],
    key=lambda x: x[0]
)
print(f"\n🏆 Best Model: {best_name} → {best_acc*100:.2f}%")

# Cross Validation
cv = cross_val_score(best_model, X_train_sc, y_train, cv=5)
print(f"\n🔁 Cross Validation: {cv.mean()*100:.2f}% ± {cv.std()*100:.2f}%")

# Classification Report
labels = ["Normal", "Moderate", "Critical"]
print(f"\n📋 Classification Report:")
print(classification_report(y_test, best_pred, target_names=labels))

# Confusion Matrix Plot
os.makedirs("model", exist_ok=True)
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle(f"{best_name} — {best_acc*100:.2f}% Accuracy")
ConfusionMatrixDisplay(confusion_matrix(y_test, best_pred),
    display_labels=labels).plot(ax=axes[0], colorbar=False, cmap="Blues")
axes[0].set_title("Confusion Matrix")

fi = best_model.feature_importances_ if hasattr(best_model,"feature_importances_") \
     else best_model.estimators_[0].feature_importances_
idx = np.argsort(fi)[::-1]
axes[1].barh([FEATURES[i] for i in idx], fi[idx], color="#2196F3")
axes[1].set_title("Feature Importance")
axes[1].invert_yaxis()
plt.tight_layout()
plt.savefig("model/training_report.png", dpi=150, bbox_inches="tight")
print("📊 Report saved → model/training_report.png")

# Save
joblib.dump(best_model, "model/pipeline_model.pkl")
joblib.dump(scaler,     "model/scaler.pkl")
joblib.dump(FEATURES,   "model/features.pkl")
print("💾 Model saved → model/pipeline_model.pkl")
print("\n✅ Training complete!")