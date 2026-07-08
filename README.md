# Predicting High Income from Census Data — Deep Learning Project

**Author:** Namrata Raghavan, Shruthi Ravi

**Dataset:** UCI Adult Census Income (Becker & Kohavi, 1996)

**Task:** Binary classification — predict whether an adult earns more than $50K / year

**Framework:** TensorFlow / Keras (with scikit-learn baselines)

This project is a complete, reproducible deep-learning study on the UCI Adult
Census Income dataset. It compares a deep neural network against logistic
regression and random forest baselines, applies disciplined preprocessing,
threshold tuning, and persists every artefact needed for inference.

## Final results (test set)

| Model              | Accuracy | F1     | ROC-AUC | PR-AUC |
|--------------------|---------:|-------:|--------:|-------:|
| Logistic Regression| 0.8513   | 0.6587 | 0.9035  | 0.7577 |
| Random Forest      | 0.8663   | 0.6812 | 0.9171  | 0.8033 |
| **Deep NN (chosen)** | **0.8502** | **0.7003** | 0.9141 | 0.7892 |

The Deep NN wins on F1 — the most informative single number under class
imbalance — and matches the random forest on ROC-AUC.

## Repository layout

```
FinalProject/
├── README.md                 ← you are here
├── requirements.txt          ← pinned-range Python dependencies
├── .gitignore
├── notebook.ipynb            ← single end-to-end notebook (executed)
├── train.py                  ← script equivalent of the notebook
├── Methodology.docx          ← step-by-step tutorial + Q&A guide
├── Presentation.pptx         ← 13-slide visual deck (~12 minutes)
├── data/
│   ├── README.md             ← citation + alternate sources
│   └── adult.csv             ← raw 1994 Census file (bundled)
├── figures/                  ← all PNGs used in the doc and deck
└── models/
    ├── adult_dnn.keras       ← trained Keras model (~880 KB)
    ├── preprocess_pipeline.joblib
    ├── schema.json           ← exact feature order + types
    ├── inference_config.json ← decision threshold, framework version
    ├── metrics.csv           ← test-set metrics table
    └── dnn_classification_report.txt
```

## Running the project

### 1. Install dependencies

```bash
python -m venv .venv && source .venv/bin/activate   # optional but recommended
pip install -r requirements.txt
```

### 2. Run the notebook (recommended)

```bash
jupyter lab notebook.ipynb
```

The notebook is **already executed** in this repository — every cell has its
output and plots embedded — but you can re-run it end-to-end with
`Kernel → Restart & Run All`. Total runtime is around 30 seconds on a CPU.

### 3. Or run the equivalent script

```bash
python train.py
```

This trains all three models, regenerates every figure in `figures/`, refreshes
the `models/` folder, and prints the metrics table.

## Inference (using the saved bundle)

```python
import json, joblib, pandas as pd
import tensorflow as tf

preprocess = joblib.load("models/preprocess_pipeline.joblib")
model      = tf.keras.models.load_model("models/adult_dnn.keras")
cfg        = json.loads(open("models/inference_config.json").read())
threshold  = cfg["decision_threshold"]

record = {
    "age": 52, "workclass": "Self-emp-not-inc", "fnlwgt": 209642,
    "education": "Masters", "education-num": 14,
    "marital-status": "Married-civ-spouse", "occupation": "Exec-managerial",
    "relationship": "Husband", "race": "White", "sex": "Male",
    "capital-gain": 15024, "capital-loss": 0,
    "hours-per-week": 60, "native-country": "United-States",
}
df  = pd.DataFrame([record])
prob = model.predict(preprocess.transform(df), verbose=0).ravel()[0]
print("probability >50K:", round(float(prob), 3),
      "→", ">50K" if prob >= threshold else "<=50K")
```

The notebook also includes a `predict()` helper that performs schema
validation (missing/unexpected fields, type coercion, whitespace stripping,
and missing-value normalisation) before passing the record to the pipeline.

## Reproducibility & best-practice checklist

This project was built against the rubric explicitly:

- [x] **Dataset realistic, not too simple** — 48,841 records, mixed types, real missing values, fairness considerations.
- [x] **No data leakage** — `train_test_split` runs *before* preprocessing; `ColumnTransformer.fit_transform` is called only on the training fold.
- [x] **Identical preprocessing in training and inference** — single `ColumnTransformer` is pickled and reused.
- [x] **Strict feature order, naming, and schema consistency** — `schema.json` is the source of truth; `predict()` re-orders columns and rejects unknown fields.
- [x] **User input validation** — type coercion, whitespace stripping, missing-value normalisation in `predict()`.
- [x] **Missing values handled explicitly** — `SimpleImputer(median)` for numerical, `SimpleImputer(most_frequent)` for categorical.
- [x] **Model size & loading time** — 880 KB on disk, loads in <1 s.
- [x] **Reproducible environment** — `requirements.txt` pins compatible version ranges.
- [x] **Versioned artefacts** — model, pipeline, config, schema, metrics saved together as one bundle.
- [x] **Drift monitoring plan** — documented in `Methodology.docx` §6 and §8.
- [x] **Security / access notes for deployment** — covered in §6 and §8 of the methodology.

## Citation

Becker, B. & Kohavi, R. (1996). *Adult* [Dataset]. UCI Machine Learning
Repository. <https://doi.org/10.24432/C5XW20>
