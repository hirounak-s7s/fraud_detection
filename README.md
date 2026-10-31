# 🛡️ Fraud Detection System

An AI-powered fraud detection system that screens mobile money transactions in real time. Built with **scikit-learn** and served through a polished **Streamlit** web interface, this project demonstrates an end-to-end machine learning pipeline — from exploratory data analysis to interactive deployment.

## How It Works

Users input transaction details — type (CASH_OUT, TRANSFER, CASH_IN, DEBIT, PAYMENT), amount, and sender/recipient balances — and receive an instant fraud verdict with probability scores. The system automatically computes derived features like balance differentials to improve detection accuracy.

## ML Pipeline

- **Model**: Logistic Regression with balanced class weights to handle the inherent class imbalance in fraud datasets
- **Preprocessing**: StandardScaler for numerical features + OneHotEncoder for categorical features, bundled into a scikit-learn Pipeline
- **Training Data**: Mobile money transaction dataset with labeled fraud/legitimate cases
- **Analysis**: Comprehensive model evaluation documented in an accompanying Jupyter notebook (`model_analysis.ipynb`)

## Tech Stack

| Component | Technology |
|---|---|
| ML Framework | scikit-learn |
| Web App | Streamlit |
| Data Processing | pandas, NumPy |
| Serialization | joblib |
| UI Design | Custom CSS with glassmorphism, gradient animations, and responsive layout |

## Features

- 🔍 Real-time transaction fraud screening
- 📊 Probability breakdown with animated confidence bars
- 🎨 Premium dark-mode UI with glassmorphism design
- 📖 Built-in feature guide and model info sidebar
- ⚡ Cached model loading for fast predictions

## Getting Started

```bash
pip install streamlit scikit-learn pandas numpy joblib
streamlit run fraud_detection.py
```

## Dataset

The training dataset (`AIML Dataset.csv`) is excluded from this repository due to its size (~494MB). Place it in the project root if you wish to retrain the model.
