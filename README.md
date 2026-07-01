# ❤️ Heart Disease Predictor

A machine learning web app that predicts a patient's risk of heart disease from clinical parameters, built with scikit-learn and deployed with Streamlit.

**🔗 Live Demo:** [heart-disease-predictor-37.streamlit.app](https://heart-disease-predictor-37.streamlit.app/)

## Overview

This project uses the UCI Heart Disease dataset (918 patient records, 11 clinical features) to train and compare multiple classification models, then deploys the best-performing model as an interactive web application for real-time risk prediction.

## Dataset

- **Source:** UCI Heart Disease dataset (`heart.csv`)
- **Records:** 918 patients
- **Features:** Age, Sex, Chest Pain Type, Resting Blood Pressure, Cholesterol, Fasting Blood Sugar, Resting ECG, Max Heart Rate, Exercise-Induced Angina, Oldpeak, ST Slope
- **Target:** Presence of heart disease (binary classification)

## Approach

1. **Exploratory Data Analysis** — distribution plots, correlation heatmap, and category-wise breakdowns to understand feature relationships with the target.
2. **Data Cleaning** — handled invalid zero values in Cholesterol and RestingBP by imputing with the feature mean; verified no missing values or duplicates.
3. **Preprocessing** — one-hot encoded categorical features and applied standard scaling to numerical features.
4. **Model Benchmarking** — trained and compared five classification algorithms:

   | Model | Accuracy | F1-Score |
   |---|---|---|
   | **Logistic Regression** | **87.5%** | **0.89** |
   | KNN | 85.9% | 0.88 |
   | SVM | 84.8% | 0.87 |
   | Naive Bayes | 84.2% | 0.86 |
   | Decision Tree | 79.4% | 0.80 |

5. **Deployment** — serialized the best model (Logistic Regression) along with the fitted scaler and feature schema, and built a Streamlit interface for real-time prediction from user-input health parameters.

## Tech Stack

- **Language:** Python
- **ML/Analysis:** scikit-learn, Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn
- **Deployment:** Streamlit
- **Model Persistence:** joblib

## Project Structure

```
Heart-Disease-Predictor/
├── Heart.ipynb            # EDA, preprocessing, model training & evaluation
├── app.py                 # Streamlit application
├── heart.csv               # Dataset
├── LogisticRegression.pkl # Trained model
├── scaler.pkl              # Fitted StandardScaler
├── columns.pkl              # Feature schema used at inference
└── requirements.txt        # Python dependencies
```

## Running Locally

```bash
git clone https://github.com/KrushnaIngale/Heart-Disease-Predictor.git
cd Heart-Disease-Predictor
pip install -r requirements.txt
streamlit run app.py
```

## Author

**Krushna Ingale**
[GitHub](https://github.com/KrushnaIngale) · [LinkedIn](https://www.linkedin.com/in/krushna-ingale)
