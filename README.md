# HomeVal AI – Intelligent House Price Estimation System

HomeVal AI is a machine learning-powered web application that estimates residential property prices using Linear Regression. The application analyzes key property features and provides accurate, data-driven price predictions through an intuitive Streamlit interface.

---

## Overview

This project demonstrates an end-to-end machine learning workflow for real estate price prediction, including:

- Data preprocessing
- Exploratory Data Analysis (EDA)
- Feature engineering
- Linear Regression model training
- Model evaluation
- Interactive Streamlit deployment

The application is designed to provide a clean, responsive, and professional user experience while showcasing practical machine learning deployment.

---

## Features

- Professional Streamlit dashboard
- Real-time house price prediction
- Interactive property input form
- Automated data preprocessing
- One-Hot Encoding for categorical features
- Linear Regression prediction pipeline
- Dataset statistics and preview
- Data visualizations
- Responsive and modern user interface
- Production-ready project structure

---

## Machine Learning Workflow

```
Dataset
    │
    ▼
Data Cleaning
    │
    ▼
Exploratory Data Analysis
    │
    ▼
Feature Preprocessing
    │
    ▼
One-Hot Encoding
    │
    ▼
Train-Test Split
    │
    ▼
Linear Regression
    │
    ▼
Model Evaluation
    │
    ▼
Model Serialization
    │
    ▼
Streamlit Deployment
```

---

## Dataset Features

| Feature | Description |
|----------|-------------|
| MSSubClass | Building Class |
| MSZoning | General Zoning Classification |
| LotArea | Lot Size |
| LotConfig | Lot Configuration |
| BldgType | Type of Dwelling |
| OverallCond | Overall Condition Rating |
| YearBuilt | Construction Year |
| YearRemodAdd | Remodel Year |
| Exterior1st | Exterior Material |
| BsmtFinSF2 | Basement Finished Area |
| TotalBsmtSF | Total Basement Area |

**Target Variable**

- SalePrice

---

## Technology Stack

### Programming Language

- Python

### Machine Learning

- Scikit-learn
- Linear Regression

### Data Processing

- Pandas
- NumPy

### Visualization

- Matplotlib
- Seaborn

### Deployment

- Streamlit

### Model Serialization

- Joblib

---

## Project Structure

```
HomeVal-AI/
│
├── app.py
├── train_model.py
├── model.pkl
├── HousePricePrediction.csv
├── requirements.txt
├── README.md
└── assets/
```

---

## Installation

Clone the repository

```bash
git clone https://github.com/your-username/HomeVal-AI.git
```

Move into the project directory

```bash
cd HomeVal-AI
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

## Train the Model

```bash
python train_model.py
```

This generates:

```
model.pkl
```

---

## Run the Application

```bash
streamlit run app.py
```

The application will launch in your default web browser.

---

## Model Information

| Property | Value |
|----------|--------|
| Algorithm | Linear Regression |
| Learning Type | Supervised Learning |
| Problem Type | Regression |
| Encoding | One-Hot Encoding |
| Framework | Scikit-learn |
| Deployment | Streamlit |

---

## Evaluation Metrics

The model is evaluated using standard regression metrics:

- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- R² Score
- Adjusted R² Score

---

## Future Improvements

- Random Forest Regression
- XGBoost Regressor
- Gradient Boosting
- Hyperparameter Optimization
- Feature Selection
- Model Explainability (SHAP)
- Cloud Deployment
- User Authentication
- Prediction History
- REST API Integration

---

## Author

**Arnav Singh**

Aspiring Machine Learning Engineer

GitHub:
https://github.com/Arnav-Singh-5080

LinkedIn:
https://www.linkedin.com/in/arnav-singh-a87847351/

Email:
itsarnav.singh80@gmail.com

---

## License

This project is developed for educational, research, and portfolio purposes.
