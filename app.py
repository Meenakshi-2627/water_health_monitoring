import streamlit as st
import numpy as np
import joblib
import matplotlib.pyplot as plt

# -------------------------
# PAGE CONFIG
# -------------------------
st.set_page_config(
    page_title="Smart Health Early Warning System",
    page_icon="🚰",
    layout="wide"
)

# -------------------------
# LOAD MODEL
# -------------------------
model = joblib.load("risk_model.pkl")
scaler = joblib.load("scaler.pkl")

# -------------------------
# HEADER
# -------------------------
st.title("🚰 Smart Community Health Monitoring System")
st.subheader("AI-Based Early Warning for Water-Borne Diseases")
st.markdown("---")

# -------------------------
# SIDEBAR INPUT
# -------------------------
st.sidebar.header("📥 Enter Environmental Parameters")

with st.sidebar.form("prediction_form"):

    rainfall = st.number_input("Rainfall (mm)", 0.0, 300.0, 50.0)
    temperature = st.number_input("Temperature (°C)", 10.0, 45.0, 28.0)
    water_quality = st.number_input("Water Quality Index", 0.0, 100.0, 70.0)

    submit = st.form_submit_button("🔍 Predict Risk")

# -------------------------
# PREDICTION
# -------------------------
if submit:

    # IMPORTANT: model must be trained with only 3 features
    input_data = np.array([[rainfall, temperature, water_quality]])
    input_scaled = scaler.transform(input_data)

    probs = model.predict_proba(input_scaled)[0]
    predicted_class = model.predict(input_scaled)[0]

    low_prob = probs[0]
    moderate_prob = probs[1]
    high_prob = probs[2]

    # -------------------------
    # DASHBOARD
    # -------------------------
    st.markdown("## 📊 Risk Probability Overview")

    col1, col2, col3 = st.columns(3)

    col1.metric("Low Risk %", f"{round(low_prob*100,2)}%")
    col2.metric("Moderate Risk %", f"{round(moderate_prob*100,2)}%")
    col3.metric("High Risk %", f"{round(high_prob*100,2)}%")

    st.markdown("---")

    # -------------------------
    # ALERT SYSTEM
    # -------------------------
    st.markdown("## 🚨 Early Warning Alert")

    if predicted_class == 2:
        st.error("🔴 HIGH RISK – Immediate Public Health Action Required")
    elif predicted_class == 1:
        st.warning("🟠 MODERATE RISK – Preventive Measures Recommended")
    else:
        st.success("🟢 LOW RISK – Situation Stable")

    st.markdown("---")

    # -------------------------
    # FEATURE IMPORTANCE
    # -------------------------
    st.markdown("## 🧠 Key Risk Factor Analysis")

    importance = model.feature_importances_
    features = ["Rainfall", "Temperature", "Water Quality"]

    fig, ax = plt.subplots()
    ax.barh(features, importance, color="#0E4C92")
    ax.set_xlabel("Importance Score")
    ax.set_title("Feature Influence on Risk Prediction")
    st.pyplot(fig)

    st.markdown("---")

    # -------------------------
    # TREND SIMULATION
    # -------------------------
    st.markdown("## 📈 30-Day Risk Trend Simulation")

    simulated_risk = np.random.randint(0, 3, 30)

    fig2, ax2 = plt.subplots()
    ax2.plot(simulated_risk, linewidth=2)
    ax2.set_xlabel("Days")
    ax2.set_ylabel("Risk Level (0=Low,1=Moderate,2=High)")
    ax2.set_title("Projected Risk Trend")
    st.pyplot(fig2)

