"""
Adult Census Income — Deep Learning Project
End-to-end training pipeline (run as a script for validation).

Author: Namrata Raghavan
"""
from __future__ import annotations

import json
import os
import random
import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_openml
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf  # noqa: E402

tf.random.set_seed(SEED)
tf.keras.utils.set_random_seed(SEED)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT = Path(__file__).resolve().parent
FIG = PROJECT / "figures"
MODELS = PROJECT / "models"
DATA = PROJECT / "data"
for p in (FIG, MODELS, DATA):
    p.mkdir(exist_ok=True)

sns.set_theme(style="whitegrid", context="talk")
plt.rcParams["figure.dpi"] = 110

# ---------------------------------------------------------------------------
# 1. Load dataset
# ---------------------------------------------------------------------------
print("[1/8] Loading UCI Adult Census Income dataset...")

ADULT_COLUMNS = [
    "age", "workclass", "fnlwgt", "education", "education-num",
    "marital-status", "occupation", "relationship", "race", "sex",
    "capital-gain", "capital-loss", "hours-per-week", "native-country",
    "income",
]

local_csv = DATA / "adult.csv"
if local_csv.exists():
    df = pd.read_csv(local_csv, header=None, names=ADULT_COLUMNS,
                     skipinitialspace=True, na_values=["?", " ?"])
else:
    # Fallback: try OpenML (requires network access to openml.org)
    ds = fetch_openml("adult", version=2, as_frame=True)
    df = ds.frame.copy()
    df.rename(columns={"class": "income"}, inplace=True)

# Normalize target -> binary  (Some sources use ">50K", others ">50K." with a dot)
target_col = "income"
df[target_col] = (df[target_col].astype(str).str.strip()
                                .str.rstrip(".")
                                .eq(">50K")).astype(int)
print(f"      shape={df.shape}  positive_rate={df[target_col].mean():.3f}")
print(f"      missing values total: {df.isna().sum().sum()}")

# ---------------------------------------------------------------------------
# 2. EDA figures
# ---------------------------------------------------------------------------
print("[2/8] Generating EDA figures...")
num_cols = df.select_dtypes(include=np.number).columns.drop(target_col).tolist()
cat_cols = df.select_dtypes(exclude=np.number).columns.tolist()

# 2a. Target distribution
fig, ax = plt.subplots(figsize=(6, 4))
counts = df[target_col].value_counts().sort_index()
ax.bar(["<=50K", ">50K"], counts.values, color=["#4C72B0", "#DD8452"])
for i, v in enumerate(counts.values):
    ax.text(i, v, f"{v:,}\n({v/len(df):.1%})", ha="center", va="bottom")
ax.set_title("Class distribution (target imbalance)")
ax.set_ylabel("Count")
fig.tight_layout()
fig.savefig(FIG / "01_class_distribution.png", bbox_inches="tight")
plt.close(fig)

# 2b. Numerical feature distributions by class
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
for ax, col in zip(axes.flat, num_cols[:6]):
    for cls, label, color in [(0, "<=50K", "#4C72B0"), (1, ">50K", "#DD8452")]:
        ax.hist(
            df.loc[df[target_col] == cls, col].dropna(),
            bins=30, alpha=0.6, label=label, color=color, density=True,
        )
    ax.set_title(col, fontsize=12)
    ax.legend(fontsize=9)
fig.suptitle("Numerical feature distributions by class", fontsize=14)
fig.tight_layout()
fig.savefig(FIG / "02_numerical_distributions.png", bbox_inches="tight")
plt.close(fig)

# 2c. Correlation heatmap
fig, ax = plt.subplots(figsize=(8, 6))
corr = df[num_cols + [target_col]].corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax,
            cbar_kws={"shrink": 0.7})
ax.set_title("Correlation heatmap (numerical features)")
fig.tight_layout()
fig.savefig(FIG / "03_correlation_heatmap.png", bbox_inches="tight")
plt.close(fig)

# 2d. Top categorical features positive-rate
fig, axes = plt.subplots(2, 2, figsize=(14, 9))
for ax, col in zip(axes.flat, ["education", "occupation", "marital-status", "sex"]):
    rates = (df.groupby(col)[target_col].mean()
               .sort_values(ascending=True).tail(12))
    ax.barh(rates.index.astype(str), rates.values, color="#55A868")
    ax.set_title(f"{col} – % earning >50K")
    ax.set_xlim(0, 1)
fig.suptitle("Positive-rate by category (top groups)", fontsize=14)
fig.tight_layout()
fig.savefig(FIG / "04_categorical_positive_rates.png", bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------------------
# 3. Train/Val/Test split BEFORE preprocessing (no leakage!)
# ---------------------------------------------------------------------------
print("[3/8] Splitting train / val / test (stratified)...")
X = df.drop(columns=[target_col])
y = df[target_col].values

X_trainval, X_test, y_trainval, y_test = train_test_split(
    X, y, test_size=0.20, random_state=SEED, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_trainval, y_trainval, test_size=0.20, random_state=SEED, stratify=y_trainval
)
print(f"      train={X_train.shape}  val={X_val.shape}  test={X_test.shape}")

# ---------------------------------------------------------------------------
# 4. Preprocessing pipeline (fit ONLY on train)
# ---------------------------------------------------------------------------
print("[4/8] Building preprocessing pipeline...")
numeric_pipeline = Pipeline([
    ("imputer", __import__("sklearn.impute", fromlist=["SimpleImputer"]).SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])
categorical_pipeline = Pipeline([
    ("imputer", __import__("sklearn.impute", fromlist=["SimpleImputer"]).SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
])
preprocess = ColumnTransformer(
    transformers=[
        ("num", numeric_pipeline, num_cols),
        ("cat", categorical_pipeline, cat_cols),
    ]
)

X_train_p = preprocess.fit_transform(X_train)
X_val_p = preprocess.transform(X_val)
X_test_p = preprocess.transform(X_test)
print(f"      processed feature dim = {X_train_p.shape[1]}")

# Save preprocessing schema
schema = {
    "numerical_features": num_cols,
    "categorical_features": cat_cols,
    "n_features_after_preprocess": int(X_train_p.shape[1]),
    "target": target_col,
    "positive_label": ">50K",
}
(MODELS / "schema.json").write_text(json.dumps(schema, indent=2))

# ---------------------------------------------------------------------------
# 5. Baseline models
# ---------------------------------------------------------------------------
print("[5/8] Training baseline models...")
results = {}

logreg = LogisticRegression(max_iter=1000, C=1.0, random_state=SEED)
logreg.fit(X_train_p, y_train)
results["LogReg"] = {
    "y_pred": logreg.predict(X_test_p),
    "y_prob": logreg.predict_proba(X_test_p)[:, 1],
}

rf = RandomForestClassifier(n_estimators=300, max_depth=18, n_jobs=-1,
                             random_state=SEED)
rf.fit(X_train_p, y_train)
results["RandomForest"] = {
    "y_pred": rf.predict(X_test_p),
    "y_prob": rf.predict_proba(X_test_p)[:, 1],
}

# ---------------------------------------------------------------------------
# 6. Deep neural network
# ---------------------------------------------------------------------------
print("[6/8] Training Deep Neural Network...")

input_dim = X_train_p.shape[1]
inputs = tf.keras.Input(shape=(input_dim,))
x = tf.keras.layers.Dense(256, activation="relu",
                          kernel_regularizer=tf.keras.regularizers.l2(1e-5))(inputs)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Dropout(0.40)(x)
x = tf.keras.layers.Dense(128, activation="relu")(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Dropout(0.30)(x)
x = tf.keras.layers.Dense(64, activation="relu")(x)
x = tf.keras.layers.Dropout(0.20)(x)
outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)
model = tf.keras.Model(inputs, outputs, name="adult_dnn")

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="binary_crossentropy",
    metrics=[
        tf.keras.metrics.BinaryAccuracy(name="acc"),
        tf.keras.metrics.AUC(name="auc"),
    ],
)
model.summary(print_fn=lambda s: print("      " + s))

callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor="val_auc", mode="max",
                                     patience=8, restore_best_weights=True,
                                     verbose=0),
    tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                         patience=3, min_lr=1e-5, verbose=0),
]

history = model.fit(
    X_train_p.astype(np.float32), y_train.astype(np.float32),
    validation_data=(X_val_p.astype(np.float32), y_val.astype(np.float32)),
    epochs=50, batch_size=256,
    callbacks=callbacks, verbose=2,
)

# Threshold tuning on validation set (maximise F1)
val_prob = model.predict(X_val_p, verbose=0).ravel()
thresholds = np.linspace(0.05, 0.95, 19)
f1_scores = [f1_score(y_val, (val_prob >= t).astype(int)) for t in thresholds]
best_t = float(thresholds[int(np.argmax(f1_scores))])
print(f"      best decision threshold (max F1 on val) = {best_t:.2f}")

dnn_prob = model.predict(X_test_p, verbose=0).ravel()
dnn_pred = (dnn_prob >= best_t).astype(int)
results["DeepNN"] = {"y_pred": dnn_pred, "y_prob": dnn_prob}

# Threshold sweep figure
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(thresholds, f1_scores, marker="o")
ax.axvline(best_t, color="red", linestyle="--",
           label=f"best={best_t:.2f}")
ax.set_xlabel("Decision threshold"); ax.set_ylabel("F1 (validation)")
ax.set_title("DNN threshold tuning"); ax.legend()
fig.tight_layout()
fig.savefig(FIG / "10_threshold_tuning.png", bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------------------
# 7. Evaluation + comparison figures
# ---------------------------------------------------------------------------
print("[7/8] Evaluating models...")
metrics_table = []
for name, r in results.items():
    metrics_table.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, r["y_pred"]),
        "Precision": precision_score(y_test, r["y_pred"]),
        "Recall": recall_score(y_test, r["y_pred"]),
        "F1": f1_score(y_test, r["y_pred"]),
        "ROC_AUC": roc_auc_score(y_test, r["y_prob"]),
        "PR_AUC": average_precision_score(y_test, r["y_prob"]),
    })
metrics_df = pd.DataFrame(metrics_table).set_index("Model").round(4)
metrics_df.to_csv(MODELS / "metrics.csv")
print("\n" + metrics_df.to_string())

# Training history
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
axes[0].plot(history.history["loss"], label="train")
axes[0].plot(history.history["val_loss"], label="val")
axes[0].set_title("DNN loss"); axes[0].set_xlabel("epoch"); axes[0].legend()
axes[1].plot(history.history["auc"], label="train")
axes[1].plot(history.history["val_auc"], label="val")
axes[1].set_title("DNN AUC"); axes[1].set_xlabel("epoch"); axes[1].legend()
fig.tight_layout()
fig.savefig(FIG / "05_training_history.png", bbox_inches="tight")
plt.close(fig)

# ROC curves
fig, ax = plt.subplots(figsize=(7, 6))
for name, r in results.items():
    fpr, tpr, _ = roc_curve(y_test, r["y_prob"])
    auc = roc_auc_score(y_test, r["y_prob"])
    ax.plot(fpr, tpr, label=f"{name}  AUC={auc:.3f}")
ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
ax.set_title("ROC curves"); ax.legend()
fig.tight_layout()
fig.savefig(FIG / "06_roc_curves.png", bbox_inches="tight")
plt.close(fig)

# PR curves
fig, ax = plt.subplots(figsize=(7, 6))
for name, r in results.items():
    p, rcl, _ = precision_recall_curve(y_test, r["y_prob"])
    ap = average_precision_score(y_test, r["y_prob"])
    ax.plot(rcl, p, label=f"{name}  AP={ap:.3f}")
ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
ax.set_title("Precision-Recall curves"); ax.legend()
fig.tight_layout()
fig.savefig(FIG / "07_pr_curves.png", bbox_inches="tight")
plt.close(fig)

# Confusion matrix for DNN
fig, ax = plt.subplots(figsize=(5, 4.5))
cm = confusion_matrix(y_test, results["DeepNN"]["y_pred"])
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
            xticklabels=["<=50K", ">50K"], yticklabels=["<=50K", ">50K"])
ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
ax.set_title("DNN confusion matrix (test)")
fig.tight_layout()
fig.savefig(FIG / "08_confusion_matrix.png", bbox_inches="tight")
plt.close(fig)

# Bar chart comparison
fig, ax = plt.subplots(figsize=(10, 5))
metrics_df[["Accuracy", "F1", "ROC_AUC", "PR_AUC"]].plot.bar(ax=ax, rot=0)
ax.set_title("Model comparison on test set")
ax.set_ylim(0.5, 1.0); ax.legend(loc="lower right", ncol=4)
fig.tight_layout()
fig.savefig(FIG / "09_model_comparison.png", bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------------------
# 8. Persist final assets
# ---------------------------------------------------------------------------
print("[8/8] Saving model + preprocessing pipeline...")
joblib.dump(preprocess, MODELS / "preprocess_pipeline.joblib")
model.save(MODELS / "adult_dnn.keras")

# Save inference threshold + run config
inference_cfg = {
    "decision_threshold": best_t,
    "feature_dim": int(X_train_p.shape[1]),
    "framework": "tensorflow-keras",
    "tf_version": tf.__version__,
    "training_seed": SEED,
}
(MODELS / "inference_config.json").write_text(json.dumps(inference_cfg, indent=2))

# Detailed classification report for the DNN (winning model)
report = classification_report(
    y_test, results["DeepNN"]["y_pred"],
    target_names=["<=50K", ">50K"], digits=4,
)
(MODELS / "dnn_classification_report.txt").write_text(report)
print("\n" + report)

print("\nDONE. Best test ROC-AUC:",
      f"{metrics_df['ROC_AUC'].max():.4f}",
      "Best test Accuracy:", f"{metrics_df['Accuracy'].max():.4f}")
