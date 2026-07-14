# Customer Churn Prediction — ANN

An end-to-end deep learning project to predict customer churn using an Artificial Neural Network (ANN), built on the Telco Customer Churn dataset.

## 📌 Overview

Telecom companies lose revenue when customers churn (leave). This project builds a binary classifier to predict whether a customer is likely to churn, based on their account and service usage details — enabling proactive retention strategies.

## 📂 Dataset

- **Source:** [Telco Customer Churn Dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) (Kaggle)
- **Rows:** ~7,000 customers
- **Target:** `Churn` (Yes/No)
- **Features:** Demographics (gender, senior citizen, partner, dependents), account info (tenure, contract, payment method), and services subscribed (internet, phone, streaming, tech support, etc.)

## 🧹 Data Preprocessing

- Converted `TotalCharges` from string to numeric, dropped rows with missing values
- Removed non-predictive `customerID` column
- Encoded target (`Churn`) and binary categorical columns (Yes/No, Male/Female) as 0/1
- One-hot encoded multi-category columns (Contract, PaymentMethod, InternetService, etc.)
- Split data into **Train (64%) / Validation (16%) / Test (20%)**
- Scaled features using `StandardScaler` (fit on train only, applied to val/test)

## 🏗️ Model Architecture

A feed-forward neural network built with TensorFlow/Keras:

```
Input Layer
Dense(32, relu)
Dropout(0.3)
Dense(16, relu)
Dropout(0.2)
Dense(1, sigmoid)
```

- **Optimizer:** Adam
- **Loss:** Binary Crossentropy
- **Regularization:** Dropout layers to reduce overfitting
- **Early Stopping:** Monitors validation loss, restores best weights
- **Class Imbalance Handling:** `class_weight="balanced"` to account for fewer churn cases than non-churn

## 📊 Evaluation

- Classification threshold tuned on the **validation set** (not test set, to avoid leakage)
- Final performance reported on a **held-out test set**, evaluated only once
- Metrics used:
  - Accuracy
  - Precision / Recall / F1-score (via `classification_report`)
  - Confusion Matrix
  - ROC-AUC
  - PR-AUC (important for imbalanced classification)

## 💾 Artifacts Saved

| File | Description |
|---|---|
| `models/customer_churn_model.keras` | Trained ANN model |
| `models/scaler.pkl` | Fitted StandardScaler for preprocessing new data |
| `models/feature_names.pkl` | Column order expected by the model |
| `models/best_threshold.pkl` | Classification threshold chosen on validation set |

## 🛠️ Tech Stack

- Python, Pandas, NumPy
- TensorFlow / Keras
- Scikit-learn
- Matplotlib

## 🚀 How to Run

1. Clone the repo and install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Update the dataset path in the notebook/script to point to your local copy of the CSV.
3. Run the notebook cells in order — from data loading through model saving.

## 📈 Key Learnings

- Avoided **test set leakage** by tuning the decision threshold on a separate validation set rather than the test set
- Handled class imbalance using class weighting instead of oversampling
- Used Dropout for regularization to improve generalization on unseen data

## 🔮 Future Improvements

- Add k-fold cross-validation for more robust performance estimates
- Try alternative architectures (deeper networks, batch normalization)
- Compare ANN performance against classical models (XGBoost, Random Forest)
- Deploy as a Streamlit demo app for interactive predictions
