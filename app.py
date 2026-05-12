"""Streamlit frontend for house price prediction."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "Housing.csv"
MODEL_PATH = BASE_DIR / "models" / "house_price_model.pkl"
METRICS_PATH = BASE_DIR / "models" / "metrics.json"

YES_NO_OPTIONS = ["yes", "no"]
FURNISHING_OPTIONS = ["furnished", "semi-furnished", "unfurnished"]


st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠",
    layout="wide",
)


@st.cache_data
def load_data() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_metrics() -> dict:
    if not METRICS_PATH.exists():
        return {}
    return json.loads(METRICS_PATH.read_text(encoding="utf-8"))


def format_rupees(value: float) -> str:
    return f"₹{value:,.0f}"


def prediction_form() -> pd.DataFrame:
    st.subheader("Enter house details")

    left, middle, right = st.columns(3)

    with left:
        area = st.number_input("Area (sq. ft.)", min_value=1650, max_value=16200, value=5000, step=50)
        bedrooms = st.number_input("Bedrooms", min_value=1, max_value=6, value=3, step=1)
        bathrooms = st.number_input("Bathrooms", min_value=1, max_value=4, value=2, step=1)
        stories = st.number_input("Stories", min_value=1, max_value=4, value=2, step=1)

    with middle:
        mainroad = st.selectbox("Main road access", YES_NO_OPTIONS)
        guestroom = st.selectbox("Guest room", YES_NO_OPTIONS, index=1)
        basement = st.selectbox("Basement", YES_NO_OPTIONS, index=1)
        hotwaterheating = st.selectbox("Hot water heating", YES_NO_OPTIONS, index=1)

    with right:
        airconditioning = st.selectbox("Air conditioning", YES_NO_OPTIONS)
        parking = st.number_input("Parking spaces", min_value=0, max_value=3, value=1, step=1)
        prefarea = st.selectbox("Preferred area", YES_NO_OPTIONS, index=1)
        furnishingstatus = st.selectbox("Furnishing status", FURNISHING_OPTIONS, index=1)

    return pd.DataFrame(
        [
            {
                "area": area,
                "bedrooms": bedrooms,
                "bathrooms": bathrooms,
                "stories": stories,
                "mainroad": mainroad,
                "guestroom": guestroom,
                "basement": basement,
                "hotwaterheating": hotwaterheating,
                "airconditioning": airconditioning,
                "parking": parking,
                "prefarea": prefarea,
                "furnishingstatus": furnishingstatus,
            }
        ]
    )


def show_prediction_tab(data: pd.DataFrame) -> None:
    if not MODEL_PATH.exists():
        st.error("Model file not found. Run `python train_model.py` first, then restart the app.")
        return

    model = load_model()
    user_input = prediction_form()

    if st.button("Predict price", type="primary", use_container_width=True):
        prediction = float(model.predict(user_input)[0])
        dataset_min = float(data["price"].min())
        dataset_max = float(data["price"].max())

        st.metric("Estimated house price", format_rupees(prediction))

        if prediction < dataset_min or prediction > dataset_max:
            st.warning(
                "This estimate is outside the training data price range, so treat it as a rough extrapolation."
            )
        else:
            st.caption(
                f"Training data price range: {format_rupees(dataset_min)} to {format_rupees(dataset_max)}"
            )

        st.dataframe(user_input, width="stretch", hide_index=True)


def show_dataset_tab(data: pd.DataFrame) -> None:
    st.subheader("Dataset overview")

    metric_cols = st.columns(4)
    metric_cols[0].metric("Rows", f"{len(data):,}")
    metric_cols[1].metric("Features", f"{len(data.columns) - 1}")
    metric_cols[2].metric("Median price", format_rupees(float(data["price"].median())))
    metric_cols[3].metric("Average area", f"{data['area'].mean():,.0f} sq. ft.")

    st.dataframe(data.head(25), width="stretch", hide_index=True)

    chart_cols = st.columns(2)
    with chart_cols[0]:
        st.caption("Price distribution")
        price_distribution = data["price"].value_counts(bins=20).sort_index()
        price_distribution.index = price_distribution.index.astype(str)
        st.bar_chart(price_distribution)
    with chart_cols[1]:
        st.caption("Average price by furnishing status")
        furnishing_summary = data.groupby("furnishingstatus")["price"].mean().sort_values()
        st.bar_chart(furnishing_summary)


def show_performance_tab(metrics: dict) -> None:
    st.subheader("Model performance")

    if not metrics:
        st.info("Metrics are not available yet. Run `python train_model.py` to generate them.")
        return

    metric_cols = st.columns(4)
    metric_cols[0].metric("Model", metrics.get("model", "Linear Regression"))
    metric_cols[1].metric("R² score", f"{metrics.get('r2_score', 0):.3f}")
    metric_cols[2].metric("RMSE", format_rupees(float(metrics.get("rmse", 0))))
    metric_cols[3].metric("MAE", format_rupees(float(metrics.get("mae", 0))))

    st.write(
        "The model is trained on an 80/20 train-test split using the same feature set from the original notebook."
    )
    st.json(metrics, expanded=False)


def show_insights_tab(data: pd.DataFrame) -> None:
    st.subheader("Feature insights")

    numeric_corr = data.select_dtypes(include="number").corr()["price"].drop("price").sort_values()

    left, right = st.columns(2)
    with left:
        st.caption("Numeric feature correlation with price")
        st.bar_chart(numeric_corr)
    with right:
        st.caption("Average price by air conditioning")
        st.bar_chart(data.groupby("airconditioning")["price"].mean().sort_values())

    st.write(
        "Area, bathrooms, stories, parking, air conditioning, preferred area, and furnishing status "
        "are useful signals in this dataset. The app uses all 12 notebook features for prediction."
    )


def main() -> None:
    st.title("House Price Prediction")
    st.caption("Portfolio-ready machine learning app using Multiple Linear Regression.")

    data = load_data()
    metrics = load_metrics()

    tabs = st.tabs(["Price prediction", "Dataset overview", "Model performance", "Feature insights"])

    with tabs[0]:
        show_prediction_tab(data)
    with tabs[1]:
        show_dataset_tab(data)
    with tabs[2]:
        show_performance_tab(metrics)
    with tabs[3]:
        show_insights_tab(data)


if __name__ == "__main__":
    main()
