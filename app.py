import streamlit as st
import numpy as np
import pickle

# =========================
# Load Model and Scaler
# =========================
with open("food_order_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# =========================
# Page Config
# =========================
st.set_page_config(
    page_title="Food Order Prediction",
    page_icon="🍔",
    layout="centered"
)

# =========================
# Title
# =========================
st.title("🍔 Food Order Prediction System")

st.write("""
This application predicts whether a user is likely to order food online
using a Decision Tree Machine Learning Model.
""")

st.markdown("---")

# =========================
# User Inputs
# =========================
st.header("Enter User Information")

# Gender
gender = st.selectbox(
    "Gender",
    ["Female", "Male"]
)
gender_val = 0 if gender == "Female" else 1

# Age Group
age_group = st.selectbox(
    "Age Group",
    ["Under 18", "18-22", "23-27", "28-32", "Above 32"]
)

age_map = {
    "Under 18": 0,
    "18-22": 1,
    "23-27": 2,
    "28-32": 3,
    "Above 32": 4
}

age_val = age_map[age_group]

# Student
student = st.selectbox(
    "Are you a student?",
    ["Yes", "No"]
)

student_val = 1 if student == "Yes" else 0

# Income
income = st.selectbox(
    "Monthly Income",
    [
        "Less than 1000",
        "1000-3000",
        "3000-5000",
        "5000-10000",
        "More than 10000"
    ]
)

income_map = {
    "Less than 1000": 0,
    "1000-3000": 1,
    "3000-5000": 2,
    "5000-10000": 3,
    "More than 10000": 4
}

income_val = income_map[income]

# Orders Per Week
orders = st.selectbox(
    "Orders Per Week",
    ["0", "1-2", "3-5", "More than 5"]
)

orders_map = {
    "0": 0,
    "1-2": 1,
    "3-5": 2,
    "More than 5": 3
}

orders_val = orders_map[orders]

# Discount Influence
discount = st.slider(
    "Discount Influence (1 = Always, 5 = Rarely)",
    1, 5, 3
)

# Delivery Fee
fee = st.selectbox(
    "Does delivery fee affect your decision?",
    ["Yes", "Sometimes", "No"]
)

fee_map = {
    "Yes": 1,
    "Sometimes": 0.5,
    "No": 0
}

fee_val = fee_map[fee]

# Ratings
ratings = st.selectbox(
    "Do restaurant ratings affect your choice?",
    ["Yes", "Sometimes", "No"]
)

ratings_map = {
    "Yes": 1,
    "Sometimes": 0.5,
    "No": 0
}

ratings_val = ratings_map[ratings]

# Reorder
reorder = st.slider(
    "Reorder from same restaurant? (1 = Always, 5 = Rarely)",
    1, 5, 3
)

# Peak Time
peak = st.selectbox(
    "Peak Ordering Time",
    ["Morning", "Afternoon", "Evening", "Late Night"]
)

peak_map = {
    "Morning": 0,
    "Afternoon": 1,
    "Evening": 2,
    "Late Night": 3
}

peak_val = peak_map[peak]

# =========================
# Delivery App
# =========================
app_used = st.selectbox(
    "Delivery App Used Most Often",
    ["Chefz", "HungerStation", "Jahez", "Keeta", "Ninja", "Other"]
)

app_chefz = 1 if app_used == "Chefz" else 0
app_hunger = 1 if app_used == "HungerStation" else 0
app_jahez = 1 if app_used == "Jahez" else 0
app_keeta = 1 if app_used == "Keeta" else 0
app_ninja = 1 if app_used == "Ninja" else 0
app_other = 1 if app_used == "Other" else 0

# =========================
# Region
# =========================
region = st.selectbox(
    "Region",
    ["Riyadh", "Jeddah", "Dammam", "Madinah", "Qassim", "Tabuk", "Other"]
)

region_dammam = 1 if region == "Dammam" else 0
region_jeddah = 1 if region == "Jeddah" else 0
region_madinah = 1 if region == "Madinah" else 0
region_other = 1 if region == "Other" else 0
region_qassim = 1 if region == "Qassim" else 0
region_riyadh = 1 if region == "Riyadh" else 0
region_tabuk = 1 if region == "Tabuk" else 0

# =========================
# Prediction
# =========================
st.markdown("---")

if st.button("Predict 🔮", use_container_width=True):

    # IMPORTANT:
    # Must match training feature order exactly

    features = np.array([[
        gender_val,
        age_val,
        student_val,
        income_val,
        orders_val,
        discount,
        fee_val,
        ratings_val,
        reorder,
        peak_val,
        app_chefz,
        app_hunger,
        app_jahez,
        app_keeta,
        app_ninja,
        app_other,
        region_dammam,
        region_jeddah,
        region_madinah,
        region_other,
        region_qassim,
        region_riyadh,
        region_tabuk
    ]])

    # Scale Features
    scaled_features = scaler.transform(features)

    # Prediction
    prediction = model.predict(scaled_features)[0]

    # Probabilities
    probabilities = model.predict_proba(scaled_features)[0]

    st.markdown("---")

    if prediction == 1:
        st.success("✅ The user is likely to order food online")

        st.metric(
            label="Confidence",
            value=f"{probabilities[1] * 100:.2f}%"
        )

    else:
        st.error("❌ The user is NOT likely to order food online")

        st.metric(
            label="Confidence",
            value=f"{probabilities[0] * 100:.2f}%"
        )

# =========================
# Footer
# =========================
st.markdown("---")
st.caption("Machine Learning Project - Decision Tree Model")
