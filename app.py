import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ---------------------------------------------------------
# Page configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="CarDekho Price Prediction",
    page_icon="🚗",
    layout="wide"
)

st.title("🚗 CarDekho Used Car Price Prediction")
st.write("Enter the car details below and the model will predict the selling price.")


# ---------------------------------------------------------
# Load and preprocess dataset
# This follows the preprocessing used in the uploaded notebook:
# - numeric missing values -> mean
# - categorical missing values -> mode
# - categorical columns -> LabelEncoder
# - target -> selling_price
# - model -> GradientBoostingRegressor
# ---------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("cardekho.csv")

    numeric_cols = df.select_dtypes(include="number").columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

    categorical_cols = df.select_dtypes(include="object").columns

    for col in categorical_cols:
        df[col] = df[col].fillna(df[col].mode()[0])

    return df


@st.cache_resource
def train_model():
    df = load_data().copy()

    categorical_cols = df.select_dtypes(include="object").columns

    # Store encoders so the same encoding used during training
    # can be applied to the user's input.
    encoders = {}

    for col in categorical_cols:
        encoder = LabelEncoder()
        df[col] = encoder.fit_transform(df[col].astype(str))
        encoders[col] = encoder

    X = df.drop("selling_price", axis=1)
    y = df["selling_price"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42
    )

    # Same Gradient Boosting model/settings as the notebook
    model = GradientBoostingRegressor(
        n_estimators=150,
        learning_rate=0.5,
        max_depth=3,
        random_state=42
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    metrics = {
        "MAE": mean_absolute_error(y_test, y_pred),
        "RMSE": mean_squared_error(y_test, y_pred) ** 0.5,
        "R2": r2_score(y_test, y_pred)
    }

    return model, encoders, metrics


df = load_data()
model, encoders, metrics = train_model()

st.success("Model loaded successfully!")


# ---------------------------------------------------------
# Model performance
# ---------------------------------------------------------
st.subheader("Model Performance")

col1, col2, col3 = st.columns(3)

col1.metric("MAE", f"₹{metrics['MAE']:,.0f}")
col2.metric("RMSE", f"₹{metrics['RMSE']:,.0f}")
col3.metric("R² Score", f"{metrics['R2']:.4f}")


# ---------------------------------------------------------
# Input form
# ---------------------------------------------------------
st.subheader("Enter Car Details")

with st.form("car_prediction_form"):

    col1, col2 = st.columns(2)

    with col1:
        name = st.selectbox(
            "Car Name",
            options=sorted(df["name"].astype(str).unique())
        )

        year = st.number_input(
            "Year",
            min_value=int(df["year"].min()),
            max_value=int(df["year"].max()),
            value=int(df["year"].median()),
            step=1
        )

        km_driven = st.number_input(
            "Kilometers Driven",
            min_value=0,
            max_value=int(df["km_driven"].max()),
            value=int(df["km_driven"].median()),
            step=100
        )

        fuel = st.selectbox(
            "Fuel Type",
            options=sorted(df["fuel"].astype(str).unique())
        )

        seller_type = st.selectbox(
            "Seller Type",
            options=sorted(df["seller_type"].astype(str).unique())
        )

        transmission = st.selectbox(
            "Transmission",
            options=sorted(df["transmission"].astype(str).unique())
        )

    with col2:
        owner = st.selectbox(
            "Owner",
            options=sorted(df["owner"].astype(str).unique())
        )

        mileage = st.number_input(
            "Mileage (km/ltr/kg)",
            min_value=float(df["mileage(km/ltr/kg)"].min()),
            max_value=float(df["mileage(km/ltr/kg)"].max()),
            value=float(df["mileage(km/ltr/kg)"].median()),
            step=0.1
        )

        engine = st.number_input(
            "Engine (CC)",
            min_value=float(df["engine"].min()),
            max_value=float(df["engine"].max()),
            value=float(df["engine"].median()),
            step=50.0
        )

        max_power = st.selectbox(
            "Max Power",
            options=sorted(df["max_power"].astype(str).unique())
        )

        seats = st.number_input(
            "Seats",
            min_value=float(df["seats"].min()),
            max_value=float(df["seats"].max()),
            value=float(df["seats"].median()),
            step=1.0
        )

    submitted = st.form_submit_button(
        "🔮 Predict Selling Price",
        use_container_width=True
    )


# ---------------------------------------------------------
# Prediction
# ---------------------------------------------------------
if submitted:

    input_data = pd.DataFrame({
        "name": [name],
        "year": [year],
        "km_driven": [km_driven],
        "fuel": [fuel],
        "seller_type": [seller_type],
        "transmission": [transmission],
        "owner": [owner],
        "mileage(km/ltr/kg)": [mileage],
        "engine": [engine],
        "max_power": [max_power],
        "seats": [seats]
    })

    # Apply exactly the same LabelEncoder mappings used during training
    for col, encoder in encoders.items():
        input_data[col] = encoder.transform(input_data[col].astype(str))

    prediction = model.predict(input_data)[0]

    st.markdown("---")
    st.subheader("💰 Predicted Selling Price")

    st.success(f"### ₹ {prediction:,.0f}")

    st.caption(
        "This prediction is generated by the Gradient Boosting Regression "
        "model trained using the uploaded CarDekho dataset."
    )


# ---------------------------------------------------------
# Dataset information
# ---------------------------------------------------------
with st.expander("View Dataset Information"):
    st.write(f"Rows: **{df.shape[0]}**")
    st.write(f"Columns: **{df.shape[1]}**")
    st.dataframe(df.head(10), use_container_width=True)
