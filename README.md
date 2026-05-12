# House Price Prediction

[![Python](https://img.shields.io/badge/Python-3.x-blue)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Deployed-ff4b4b)](https://housing-price-prediction-s.streamlit.app/)
[![Scikit-learn](https://img.shields.io/badge/ML-Scikit--learn-orange)](https://scikit-learn.org/)

A machine learning web application that predicts house prices from property details such as area, bedrooms, bathrooms, parking, furnishing status, and key amenities. The project combines exploratory data analysis, a reusable scikit-learn training pipeline, and a deployed Streamlit frontend.

## Live Demo

Try the deployed app here:

**https://housing-price-prediction-s.streamlit.app/**

## Project Highlights

- Built an end-to-end regression workflow for house price prediction.
- Trained a Multiple Linear Regression model using 12 housing features.
- Added preprocessing with one-hot encoding for categorical inputs.
- Saved the trained model as a reusable `joblib` artifact.
- Built an interactive Streamlit frontend for real-time predictions.
- Included dataset overview, model performance, and feature insight sections.
- Deployed the application on Streamlit Community Cloud.

## Application Preview

The Streamlit app includes:

- A prediction form for entering house details.
- Estimated house price output formatted in Indian rupees.
- Dataset summary and sample records.
- Model performance metrics including RMSE, MAE, and R2 score.
- Feature insight charts for understanding price relationships.

## Tech Stack

| Area | Tools |
| --- | --- |
| Language | Python |
| Data Processing | Pandas, NumPy |
| Machine Learning | Scikit-learn |
| Visualization | Streamlit charts, Matplotlib, Seaborn |
| Model Persistence | Joblib |
| Deployment | Streamlit Community Cloud |

## Model Performance

The current model is trained with an 80/20 train-test split.

| Metric | Value |
| --- | ---: |
| Model | Multiple Linear Regression |
| Training Rows | 436 |
| Testing Rows | 109 |
| RMSE | 1,324,506.96 |
| MAE | 970,043.40 |
| R2 Score | 0.6529 |

## Dataset Features

The model uses the following input features:

| Feature | Description |
| --- | --- |
| `area` | House area in square feet |
| `bedrooms` | Number of bedrooms |
| `bathrooms` | Number of bathrooms |
| `stories` | Number of floors/stories |
| `mainroad` | Whether the house is connected to the main road |
| `guestroom` | Guest room availability |
| `basement` | Basement availability |
| `hotwaterheating` | Hot water heating availability |
| `airconditioning` | Air conditioning availability |
| `parking` | Number of parking spaces |
| `prefarea` | Whether the house is in a preferred area |
| `furnishingstatus` | Furnished, semi-furnished, or unfurnished |

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

## Local Setup

Clone the repository and install dependencies:

```bash
git clone <repository-url>
cd Housing-Price-Prediction

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Train the model:

```bash
python train_model.py
```

Run the Streamlit app locally:

```bash
streamlit run app.py
```

## Workflow

1. Load and inspect the housing dataset.
2. Select numeric and categorical housing features.
3. Encode categorical columns with `OneHotEncoder`.
4. Train a Multiple Linear Regression model.
5. Save the model pipeline and metrics.
6. Load the saved model in Streamlit.
7. Accept user inputs and generate a price prediction.

## Deployment

The app is deployed on Streamlit Community Cloud:

```text
https://housing-price-prediction-s.streamlit.app/
```

For deployment, Streamlit uses:

- `app.py` as the main application file.
- `requirements.txt` for Python dependencies.
- `models/house_price_model.pkl` and `models/metrics.json` for prediction and metrics display.

## Future Improvements

- Compare Linear Regression with Ridge, Lasso, Random Forest, and XGBoost.
- Add cross-validation for more robust model evaluation.
- Improve prediction reliability with outlier handling.
- Add feature engineering such as price per square foot and amenity score.
- Add screenshots or GIFs of the deployed application.

## Author

**Sumit Purnapatre**
