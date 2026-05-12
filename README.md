# House Price Prediction

A portfolio-ready machine learning project that predicts house prices from property features such as area, bedrooms, bathrooms, amenities, parking, and furnishing status.

The original notebook contains the exploratory analysis and baseline model work. The Streamlit frontend adds an interactive prediction experience powered by a saved scikit-learn pipeline.

## Features

- Interactive Streamlit prediction form
- Saved scikit-learn model pipeline
- One-hot encoding for categorical housing features
- Model metrics stored in JSON
- Dataset overview and feature insight tabs
- Indian rupee formatting for predicted prices

## Tech Stack

- Python
- Pandas and NumPy
- Scikit-learn
- Streamlit
- Matplotlib and Seaborn
- Joblib

## Project Structure

```text
.
├── HOUSING_PRICE_CLEANED.ipynb
├── Housing.csv
├── app.py
├── train_model.py
├── requirements.txt
├── models/
│   ├── house_price_model.pkl
│   └── metrics.json
└── README.md
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Train the Model

Run this once before starting the frontend:

```bash
python train_model.py
```

This creates:

- `models/house_price_model.pkl`
- `models/metrics.json`

## Run the Frontend

```bash
streamlit run app.py
```

The app opens with the prediction form first. Enter house details and click **Predict price** to get an estimated house price.

## Model Inputs

The app uses the same 12 input features as the notebook:

- `area`
- `bedrooms`
- `bathrooms`
- `stories`
- `mainroad`
- `guestroom`
- `basement`
- `hotwaterheating`
- `airconditioning`
- `parking`
- `prefarea`
- `furnishingstatus`

## Model

- Algorithm: Multiple Linear Regression
- Split: 80% training, 20% testing
- Encoding: One-hot encoding for categorical fields
- Metrics: RMSE, MAE, and R² score

## Screenshots

Add screenshots here after running the Streamlit app locally.

## Author

Sumit Purnapatre
