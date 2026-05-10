import streamlit as st
import numpy as np
import pickle

# Load the saved model and scaler
with open('food_order_model.pkl', 'wb') as f:
    model = pickle.load(f)

with open('scaler.pkl', 'wb') as f:
    scaler = pickle.load(f)

# Page configuration
st.set_page_config(page_title="Food Order Prediction", page_icon="🍔", layout="centered")

# Title
st.title("🍔 Food Order Prediction App")
st.write("This app predicts whether a user is likely to order food online based on their behavior and preferences.")

st.markdown("---")

# Input fields
st.header("Please enter your information:")

# 1. Gender
gender = st.selectbox("Gender", ["Female", "Male"])
gender_val = 0 if gender == "Female" else 1

# 2. Age Group
age_group = st.selectbox("Age Group", ["Under 18", "18-22", "23-27", "28-32", "Above 32"])
age_map = {"Under 18": 0, "18-22": 1, "23-27": 2, "28-32": 3, "Above 32": 4}
age_val = age_map[age_group]

# 3. Student
is_student = st.selectbox("Are you currently a student?", ["Yes", "No"])
student_val = 1 if is_student == "Yes" else 0

# 4. Income Level
income = st.selectbox("Monthly Income Level", 
                     ["Less than 1000", "1000-3000", "3000-5000", "5000-10000", "More than 10000"])
income_map = {"Less than 1000": 0, "1000-3000": 1, "3000-5000": 2, "5000-10000": 3, "More than 10000": 4}
income_val = income_map[income]

# 5. Orders per week
orders = st.selectbox("Number of Delivery Orders per Week", 
                     ["0", "1-2", "3-5", "More than 5"])
orders_map = {"0": 0, "1-2": 1, "3-5": 2, "More than 5": 3}
orders_val = orders_map[orders]

# 6. Discount Influence
discount = st.slider("Do discounts and promotions influence your order? (1=Always, 5=Rarely)", 1, 5, 3)

# 7. Delivery Fee Effect
fee_effect = st.selectbox("Does the delivery fee affect your decision?", ["Yes", "No", "Sometimes"])
fee_map = {"Yes": 1.0, "No": 0.0, "Sometimes": 0.5}
fee_val = fee_map[fee_effect]

# 8. Relies on Ratings
ratings = st.selectbox("Do you rely on restaurant ratings?", ["Yes", "No", "Sometimes"])
ratings_map = {"Yes": 1.0, "No": 0.0, "Sometimes": 0.5}
ratings_val = ratings_map[ratings]

# 9. Reorder Tendency
reorder = st.slider("Do you tend to reorder from the same restaurant? (1=Always, 5=Rarely)", 1, 5, 3)

# 10. Peak Time
peak = st.selectbox("Peak Ordering Time", 
                   ["Morning", "Afternoon", "Evening", "Late Night"])
peak_map = {"Morning": 0, "Afternoon": 1, "Evening": 2, "Late Night": 3}
peak_val = peak_map[peak]

# 11. Delivery App
app_used = st.selectbox("Delivery App Used Most Often", 
                       ["Chefz", "HungerStation", "Jahez", "Keeta", "Ninja", "Other"])
app_chefz = 1 if app_used == "Chefz" else 0
app_hunger = 1 if app_used == "HungerStation" else 0
app_jahez = 1 if app_used == "Jahez" else 0
app_keeta = 1 if app_used == "Keeta" else 0
app_ninja = 1 if app_used == "Ninja" else 0
app_other = 1 if app_used == "Other" else 0

# 12. Region
region = st.selectbox("Region", 
                     ["Riyadh", "Jeddah", "Dammam", "Madinah", "Qassim", "Tabuk", "Other"])
region_dammam = 1 if region == "Dammam" else 0
region_jeddah = 1 if region == "Jeddah" else 0
region_madinah = 1 if region == "Madinah" else 0
region_other = 1 if region == "Other" else 0
region_qassim = 1 if region == "Qassim" else 0
region_riyadh = 1 if region == "Riyadh" else 0
region_tabuk = 1 if region == "Tabuk" else 0

st.markdown("---")

# Prediction button
if st.button("Predict 🔮", use_container_width=True):
    # Build the feature array (must match training order!)
    features = np.array([[
        gender_val, age_val, student_val, income_val, orders_val,
        discount, fee_val, ratings_val, reorder, peak_val,
        app_chefz, app_hunger, app_jahez, app_keeta, app_ninja, app_other,
        region_dammam, region_jeddah, region_madinah, region_other,
        region_qassim, region_riyadh, region_tabuk
    ]])
    
    # Scale features
    features_scaled = scaler.transform(features)
    
    # Predict
    prediction = model.predict(features_scaled)[0]
    probability = model.predict_proba(features_scaled)[0]
    
    # Display result
    if prediction == 1:
        st.success("✅ **Will Order** food online")
        st.write(f"Confidence: **{probability[1]*100:.2f}%**")
    else:
        st.warning("❌ **Will Not Order** food online")
        st.write(f"Confidence: **{probability[0]*100:.2f}%**")

st.markdown("---")
st.caption("Machine Learning Project - DS 323")
