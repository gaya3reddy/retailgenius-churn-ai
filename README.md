# RetailGenius – AI-Powered Customer Churn Prediction

This project implements an **end-to-end machine learning pipeline** for customer churn prediction in an e-commerce context.
It was developed as part of the **EPITA – AI Project Methodology course**, covering both **functional (Part 1)** and **technical (Part 2)** aspects of an AI project.

The focus of this project is **methodology, reproducibility, and engineering best practices**, rather than maximizing predictive performance.

---

## 🎯 Project Objectives

* Predict customer churn based on historical behavioral and transactional data
* Design a **reproducible and modular ML pipeline**
* Apply **software engineering best practices** to an AI project
* Track experiments and models using **MLflow**
* Provide a clean **training and inference workflow**

---

## 🏗️ Project Structure

The repository follows a production-oriented structure inspired by **Cookiecutter Data Science**:

```
retailgenius-churn-ai/
│
├── data/
│   ├── raw/                # Raw input data (not versioned)
│   ├── processed/          # Cleaned data, features, predictions
│
├── notebooks/
│   └── 01_eda.ipynb        # Exploratory Data Analysis
│
├── src/
│   ├── data/
│   │   └── make_dataset.py         # Data ingestion & basic cleaning
│   ├── features/
│   │   └── build_features.py       # Feature engineering
│   ├── models/
│   │   ├── train_model.py          # Model training + MLflow logging
│   │   └── predict_model.py        # Model inference using MLflow
│   └── visualization/
│       └── visualize.py            # (Optional) visualizations
│
├── mlruns/                 # MLflow experiments (ignored in Git)
├── requirements.txt        # Project dependencies
├── README.md               # Project documentation
└── .gitignore
```

---

## 📊 Data & Feature Engineering

* Raw e-commerce data is ingested from Excel/CSV files
* Data cleaning includes:

  * Removal of duplicates
  * Handling missing values
* Feature engineering is implemented via scripts to ensure reproducibility
* Final features are stored in `data/processed/features.csv`

---

## 🤖 Model Training

* **Baseline model:** Logistic Regression
* **Why Logistic Regression?**

  * Interpretable
  * Robust baseline for churn prediction
  * Well-suited for imbalanced classification problems

### Training Pipeline

A **Scikit-learn Pipeline** is used to avoid data leakage:

* Median imputation for missing values
* Feature scaling
* Model training

---

## 📈 Experiment Tracking with MLflow

MLflow is used to manage experiments and models:

* Track parameters and metrics:

  * Accuracy
  * F1-score
  * ROC-AUC
* Log model artifacts
* Store model signatures and input examples
* Compare multiple runs through the MLflow UI

Each experiment is logged under the experiment name:

```
RetailGenius-Churn
```

---

## 🔮 Prediction Pipeline

A dedicated inference script (`predict_model.py`) allows predictions on new data:

* Loads the trained model directly from MLflow using a `runs:/` URI
* Applies the same preprocessing pipeline used during training
* Outputs predictions as a CSV file

Example usage:

```bash
python src/models/predict_model.py \
  --model-uri runs:/<RUN_ID>/model \
  --input data/processed/features.csv \
  --output data/processed/predictions.csv
```

---

## 🧪 Code Quality & Reproducibility

* Python dependencies are managed via a virtual environment and `requirements.txt`
* Static code analysis performed using **flake8**
* Data artifacts and MLflow runs are excluded from version control
* GitHub is used for:

  * Version control
  * Feature branches
  * Clean, incremental commits

---

## 📌 Key Technologies

* Python
* Pandas, NumPy
* Scikit-learn
* MLflow
* Git & GitHub
* flake8

---

## 📝 Academic Context

This project was developed for:

* **Course:** AI Project Methodology
* **Institution:** EPITA
* **Scope:**

  * Part 1: Functional methodology (business understanding, governance)
  * Part 2: Technical methodology (implementation & MLOps practices)

---

Great idea — this is the **last missing piece** that makes your README feel *complete and professional*.
Below is a **clear, beginner-proof “How to run from scratch” section** that matches your project exactly.

You can copy-paste this **as-is** under a new section in your `README.md`.

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

RetailGenius demonstrates a **complete AI project lifecycle**, from data preparation to model training, experiment tracking, and inference.
The project emphasizes **engineering discipline, reproducibility, and clarity**, aligning with real-world AI system development practices.

