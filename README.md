
# RetailGenius – AI-Powered Customer Churn Prediction (XGBoost + SHAP)

An **end-to-end, production-style machine learning system** for customer churn prediction in an e-commerce context.

Developed as part of the **EPITA – AI Project Methodology** course, this project demonstrates the complete AI lifecycle:
from business understanding and data engineering to model training, MLOps, explainability, and deployment.

---

## 🎯 Project Objectives

- Predict customer churn using behavioral and transactional features  
- Build a **modular and reproducible ML pipeline**  
- Track experiments and models with **MLflow**  
- Apply **XGBoost** for high-performance learning  
- Add **SHAP explainability (XAI)** for model transparency  
- Register and serve models using **MLflow Model Registry**  
- Follow **CRISP-DM** and MLOps best practices  

---

## 🏗️ Project Structure

```

retailgenius-churn-ai/
│
├── data/
│   ├── raw/
│   ├── processed/
│
├── notebooks/
│   └── 01_eda.ipynb
│
├── reports/
│   └── xai_outputs/
│       ├── shap_beeswarm.png
│       ├── shap_summary_bar.png
│       ├── shap_dependence_top_feature.png
│       └── shap_waterfall_row_0.png
│
├── src/
│   ├── data/
│   │   └── make_dataset.py
│   ├── features/
│   │   └── build_features.py
│   ├── models/
│   │   ├── train_model.py
│   │   ├── train_xgb_model.py
│   │   └── predict_model.py
│   ├── xai/
│       └── shap_explain.py
│
├── docs/
├── mlruns/
├── requirements.txt
├── README.md
└── .gitignore

```

---

## 📊 Data & Features

- Dataset: Kaggle E-Commerce Customer Churn Dataset  
- Cleaning:
  - Duplicate removal  
  - Missing value imputation  
- Feature engineering:
  - Behavioral frequency
  - Order patterns
  - Customer profile attributes  

Final features are stored in:

```

data/processed/features.csv

```

---

## 🤖 Models

| Model | Purpose | Notes |
|------|--------|------|
| Logistic Regression | Baseline | Interpretable, fast |
| XGBoost (Advanced) | Production model | High performance, imbalance-aware |

### Imbalance Handling

```

Churn = 0 → 4682
Churn = 1 → 948

````

Handled using:

```python
scale_pos_weight = negative / positive
````

in XGBoost.

---

## 📈 MLflow Tracking & Registry

* Parameters, metrics, and artifacts are logged
* Models are registered in the **MLflow Model Registry**
* Each run stores:

  * Accuracy
  * F1-score
  * ROC-AUC
  * Model signature & input examples

### Launch MLflow UI

```bash
mlflow ui --host 127.0.0.1 --port 5000
```

Open:

```
http://127.0.0.1:5000
```

---

## 🔍 Explainable AI with SHAP

SHAP is applied to the **XGBoost model**:

* Global feature importance (beeswarm & bar plots)
* Feature interaction analysis
* Local explanations (waterfall plots)

Saved in:

```
reports/xai_outputs/
```

---

## ▶️ How to Run the Project from Scratch

### 1. Clone

```bash
git clone https://github.com/gaya3reddy/retailgenius-churn-ai.git
cd retailgenius-churn-ai
```

### 2. Create Environment

```bash
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install

```bash
pip install -r requirements.txt
```

### 4. Add Raw Data

```
data/raw/E_Commerce_Dataset.xlsx
```

### 5. Prepare Data

```bash
python src/data/make_dataset.py
```

### 6. Build Features

```bash
python src/features/build_features.py
```

### 7. Train XGBoost

```bash
python src/models/train_xgb_model.py
```

### 8. Explain with SHAP

```bash
python src/xai/shap_explain.py
```

### 9. Predict

```bash
python src/models/predict_model.py \
  --model-uri models:/RetailGenius_Churn_Model/Production \
  --input data/processed/features.csv \
  --output data/processed/predictions.csv
```

---

## 🧪 Code Quality

```bash
flake8 src
```

---

## 📌 Key Technologies

* Python
* Pandas, NumPy
* Scikit-learn
* XGBoost
* MLflow (Tracking + Registry + Serving)
* SHAP (Explainable AI)
* Sphinx
* Git & GitHub
* flake8

---

## 📝 Academic Context

**Course:** AI Project Methodology
**Institution:** EPITA

**Scope:**

* Part 1 – CRISP-DM, business, governance
* Part 2 – ML pipeline & MLOps
* Part 3 – XAI, registry, deployment

---

## ▶️ How to Run the Project from Scratch

This section explains how to set up the environment and run the complete pipeline from raw data to predictions.

---

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/gaya3reddy/retailgenius-churn-ai.git
cd retailgenius-churn-ai
```

---

### 2️⃣ Create and Activate a Virtual Environment

**Windows (PowerShell):**

```bash
python -m venv .venv
.venv\Scripts\activate
```

**macOS / Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

---

### 3️⃣ Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

### 4️⃣ Prepare the Data

Place the raw dataset file in:

```
data/raw/
```

Example:

```
data/raw/E_Commerce_Dataset.xlsx
```

---

### 5️⃣ Run Data Preparation

This step cleans the raw data and saves a processed dataset.

```bash
python src/data/make_dataset.py
```

Output:

```
data/processed/processed_churn.csv
```

---

### 6️⃣ Run Feature Engineering

This step generates model-ready features.

```bash
python src/features/build_features.py
```

Output:

```
data/processed/features.csv
```

---

### 7️⃣ Train the Model and Track Experiments

This step:

* Trains the baseline model
* Logs metrics and artifacts using MLflow

```bash
python src/models/train_model.py
```

---

### 8️⃣ Launch MLflow UI (Optional but Recommended)

```bash
mlflow ui --host 127.0.0.1 --port 5000
```

Open in browser:

```
http://127.0.0.1:5000
```

---

### 9️⃣ Generate Predictions

Use the trained model to generate predictions on new data.

```bash
python src/models/predict_model.py \
  --model-uri runs:/<RUN_ID>/model \
  --input data/processed/features.csv \
  --output data/processed/predictions.csv
```

Output:

```
data/processed/predictions.csv
```

---

### 🔁 Notes on Reproducibility

* All preprocessing steps are embedded in a Scikit-learn Pipeline
* The same pipeline is used for training and inference
* MLflow ensures experiment and model reproducibility
* Data and model artifacts are excluded from version control

---

### 🧪 Code Quality Check (Optional)

```bash
flake8 src
```

---


## ✅ Conclusion

RetailGenius demonstrates a **real-world AI system**, combining:

* Engineering discipline
* Explainable AI
* Reproducibility
* Deployment readiness

This project reflects **industry-grade ML workflows** rather than a single model experiment.

```

---


