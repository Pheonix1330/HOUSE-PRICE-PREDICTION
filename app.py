import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="House Price Prediction",page_icon="🏠",layout="centered")

MODEL_PATH = "MODELS/random_fores_model1.pkl"
SCALER_PATH = "MODELS/scaler1.pkl"

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

st.title("🏠 House Price Prediction")

st.write(
    "Enter the values for the three positively correlated "
    "features to predict the house price."
)

st.subheader("Enter House Details")

MedInc = st.number_input("Median Income (MedInc)",min_value=0.0,value=3.0,step=0.1,help="Median household income.")

AveRooms = st.number_input("Average Rooms (AveRooms)",min_value=0.0,value=5.0,step=0.1,help="Average number of rooms per household.")

HouseAge = st.number_input("House Age (HouseAge)",min_value=0.0,value=30.0,step=1.0,help="Age of the house in years.")

if st.button("Predict House Price", type="primary"):
    input_data = pd.DataFrame({
        "MedInc": [MedInc],
        "AveRooms": [AveRooms],
        "HouseAge": [HouseAge]
    })

    # Scale input using the same scaler used during training
    input_scaled = scaler.transform(input_data)

    # Prediction
    prediction = model.predict(input_scaled)[0]

    st.success("Prediction Completed")

    st.subheader("Predicted House Price")

    st.metric(
        label="Estimated Price",
        value=f"${prediction:,.2f}"
    )