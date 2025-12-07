# Credit Card Fraud Detection (ML Project)

End-to-end machine learning project for credit card fraud detection using the [Kaggle credit card fraud dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).  
Includes data preprocessing, model training and comparison, cost-sensitive evaluation, and an interactive Streamlit app.

---

## Project Overview

This repository contains the code of my final degree project (TFG) focused on **fraud detection in credit card transactions**.

Key aspects:

- Highly **imbalanced dataset** (legitimate vs fraudulent transactions).
- **Supervised machine learning** for fraud detection.
- Comparison between different models and configurations.
- **Cost-sensitive evaluation**: focus on the financial impact of false positives / false negatives.
- Simple **web interface with Streamlit** to interact with the trained model.

---

## Main Features

- Data loading and preprocessing.
- Baseline Random Forest model.
- Optimized Random Forest model (hyperparameter tuning).
- Model comparison using standard metrics (accuracy, precision, recall, F1, ROC AUC, etc.).
- Cost-based evaluation of models.
- Streamlit app for demo and manual testing of fraud detection.

---

## Project Structure

Approximate structure of the repository:

```text
CreditCard-FraudDetection/
├─ app_streamlit.py          # Streamlit web app
├─ core.py                   # Core utilities (data loading, preprocessing, helpers)
├─ train_rf_baseline.py      # Training script for baseline Random Forest
├─ train_rf_optimized.py     # Training script for optimized Random Forest
├─ compare_models.py         # Model comparison using multiple metrics
├─ compare_models_cost.py    # Cost-sensitive model evaluation
├─ verificar_duracion.py     # Small utility script (e.g. timing / checks)
├─ models/                   # Trained models (.pkl) tracked with Git LFS
├─ data/                     # Local data folder (not versioned: creditcard.csv)
├─ README.md                 # Project documentation
├─ .gitignore
└─ .gitattributes

```

Note:
The original dataset file creditcard.csv is not included in the repository.
Trained models (.pkl) are tracked using Git LFS.

⸻

Tech Stack
	•	Language: Python
	•	Data & ML: pandas, NumPy, scikit-learn
	•	Visualisation (optional): matplotlib / seaborn
	•	Web app: Streamlit
	•	Version control: Git + GitHub (Git LFS for model files)

⸻

Dataset

The project uses the public dataset:

Credit Card Fraud Detection – Kaggle
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

Steps to download:
	1.	Go to the dataset page on Kaggle.
	2.	Log in with your Kaggle account.
	3.	Download the file creditcard.csv.
	4.	Place it under the project directory, for example:
```
CreditCard-FraudDetection/
└─ data/
   └─ creditcard.csv
```
Make sure that the path used in the scripts (e.g. in core.py) matches the location of creditcard.csv.

⸻

Setup
	1.	Clone the repository:

```
git clone https://github.com/MarcosIbanezFandos/CreditCard-FraudDetection.git
cd CreditCard-FraudDetection
```

  2.	Create and activate a virtual environment (optional but recommended):
```
python -m venv .venv
source .venv/bin/activate  # Mac / Linux
# .venv\Scripts\activate   # Windows (if applicable)
```
  3.	Install dependencies:

If there is a requirements.txt file:
```
pip install -r requirements.txt
```
Otherwise, install the main libraries manually (example):
```
pip install pandas numpy scikit-learn streamlit matplotlib seaborn
```
  4.	Download the dataset from Kaggle as explained above and place creditcard.csv in data/.

⸻

Training the Models

You can train the models from the command line.

Baseline Random Forest
```
python train_rf_baseline.py
```
Optimized Random Forest
```
python train_rf_optimized.py
```
The scripts will:
	•	Load creditcard.csv from the configured path (e.g. data/creditcard.csv).
	•	Train the corresponding model.
	•	Save the trained model as a .pkl file under the models/ folder.

⸻

Model Evaluation & Cost Analysis

To compare the performance of different models:
```
python compare_models.py
```
To run the cost-sensitive evaluation:
```
python compare_models_cost.py
```
These scripts generate metrics and (optionally) plots to analyse which model is more suitable for fraud detection under different constraints.

⸻

Streamlit App

The Streamlit app allows you to interactively test the fraud detection model (for example, by loading transactions or manually setting feature values).

Run:
```
streamlit run app_streamlit.py
```
Then open the local URL shown in the terminal (usually http://localhost:8501) in your browser.

⸻

Notes on Git LFS

Trained models (.pkl files) can be quite large.
For that reason, they are tracked with Git LFS.

If you clone this repository and want to pull the model files, make sure you have Git LFS installed:
```
git lfs install
git lfs pull
```
If you prefer, you can also retrain the models locally using the training scripts instead of downloading the .pkl files.

⸻

Contact

Author: Marcos Ibáñez Fandos
GitHub: @MarcosIbanezFandos
Email: marcosibanezfandos@gmail.com

Feel free to reach out if you have any questions about the project or want to discuss machine learning and fraud detection in fintech.
