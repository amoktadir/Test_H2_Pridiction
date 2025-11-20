import streamlit as st
import pickle
import numpy as np
import os

# Set page configuration
st.set_page_config(
    page_title="SCWG Hydrogen Production Predictor",
    page_icon="⚡",
    layout="wide"
)

# Load the trained model and scaler
@st.cache_resource
def load_model():
    try:
        with open('rf_model.pkl', 'rb') as file:
            model = pickle.load(file)
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

@st.cache_resource
def load_scaler():
    try:
        with open('scaler.pkl', 'rb') as file:
            scaler = pickle.load(file)
        return scaler
    except Exception as e:
        st.error(f"Error loading scaler: {str(e)}")
        return None

# Load models
model = load_model()
scaler = load_scaler()

# Carbon reduction factors (kgCO2e per tonne of waste treated via SCWG instead of land disposal)
CARBON_REDUCTION_FACTORS = {
    'Sewage Sludge': 310,      # Average of 150-550 kgCO2e/tonne
    'Lignocellulosic Biomass': 400,  # Average of 150-450 kgCO2e/tonne
    'Petrochemical': 240        # Average of 400-900 kgCO2e/tonne
}

# Conversion factors
TREE_SEQUESTRATION_FACTOR = 25  # kg CO2/tree/year
CAR_EMISSIONS_FACTOR = 0.250    # kg CO2/km for 1.8L gasoline car
BLUE_H2_SAVINGS_FACTOR = 1.6    # kg CO2 saved per kg H2 compared to gasoline

# App title and description
st.title("⚡ SCWG Hydrogen Production Predictor")
st.markdown("""
This app predicts hydrogen production from Supercritical Water Gasification (SCWG) of various waste materials.
Enter the parameters below to get predictions.
""")

# Create input form
with st.form("prediction_form"):
    st.header("Input Parameters")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Waste Composition (%)")
        C = st.number_input("Carbon (C)", min_value=0.0, max_value=100.0, value=50.0, step=0.1)
        H = st.number_input("Hydrogen (H)", min_value=0.0, max_value=100.0, value=6.0, step=0.1)
        N = st.number_input("Nitrogen (N)", min_value=0.0, max_value=100.0, value=2.0, step=0.1)
        O = st.number_input("Oxygen (O)", min_value=0.0, max_value=100.0, value=30.0, step=0.1)
        
        # Calculate and display ultimate analysis sum
        ultimate_sum = C + H + N + O
        st.info(f"Ultimate Analysis Sum: {ultimate_sum:.1f}%")
        if ultimate_sum > 100:
            st.error("⚠️ Ultimate Analysis sum cannot exceed 100%")
    
    with col2:
        st.subheader("Process Conditions")
        SC = st.number_input("Solid Content (%)", min_value=0.1, max_value=99.9, value=15.0, step=0.1)
        TEMP = st.slider("Temperature (°C)", min_value=300, max_value=650, value=500)
        P = st.slider("Pressure (MPa)", min_value=10, max_value=35, value=25)
        RT = st.number_input("Reaction Time (min)", min_value=0.0, value=30.0, step=1.0)
        
        st.subheader("Waste Details")
        waste_amount = st.number_input("Waste Amount (kg)", min_value=0.1, value=100.0, step=1.0)
        waste_type = st.selectbox(
            "Waste Type",
            options=list(CARBON_REDUCTION_FACTORS.keys()),
            index=0
        )
    
    # Submit button
    submitted = st.form_submit_button("Predict Hydrogen Production")

# Handle prediction
if submitted:
    if model is None or scaler is None:
        st.error("❌ Model or scaler not loaded properly. Please check your .pkl files.")
    else:
        # Validate inputs
        if ultimate_sum > 100:
            st.error("Please adjust the Ultimate Analysis values (sum must be ≤100%)")
        elif SC >= 100:
            st.error("Solid Content must be less than 100%")
        elif waste_amount <= 0:
            st.error("Waste amount must be greater than 0 kg")
        else:
            try:
                # Prepare features for prediction
                features = [C, H, N, O, SC, TEMP, P, RT]
                features_array = np.array(features).reshape(1, -1)
                
                # Scale the features
                features_scaled = scaler.transform(features_array)
                
                # Make prediction
                with st.spinner('Making prediction...'):
                    h2_yield = model.predict(features_scaled)[0]
                
                # Calculate results
                total_h2_mol = h2_yield * waste_amount
                total_h2_kg = total_h2_mol * 0.002016  # 1 mole H2 = 0.002016 kg
                
                # Calculate CO2e reduction
                waste_amount_tonnes = waste_amount / 1000
                carbon_reduction = CARBON_REDUCTION_FACTORS.get(waste_type, 0) * waste_amount_tonnes
                carbon_sequestration = carbon_reduction / TREE_SEQUESTRATION_FACTOR
                car_travel_km = carbon_reduction / CAR_EMISSIONS_FACTOR
                co2_saved_h2 = total_h2_kg * BLUE_H2_SAVINGS_FACTOR
                
                # Display results
                st.success("✅ Prediction completed successfully!")
                
                # Results in columns
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("H₂ Yield", f"{h2_yield:.2f} mol/kg")
                    st.metric("Total H₂ Production", f"{total_h2_kg:.2f} kg")
                    st.metric("Total H₂ Moles", f"{total_h2_mol:.2f} mol")
                
                with col2:
                    st.metric("CO₂ Reduction", f"{carbon_reduction:.2f} kgCO₂e")
                    st.metric("Equivalent Tree Sequestration", f"{carbon_sequestration:.1f} tree-years")
                
                with col3:
                    st.metric("Equivalent Car Travel", f"{car_travel_km:.1f} km")
                    st.metric("CO₂ Saved vs Blue H₂", f"{co2_saved_h2:.2f} kgCO₂e")
                
                # Additional information
                with st.expander("📊 Detailed Information"):
                    st.write(f"**Waste Type:** {waste_type}")
                    st.write(f"**Waste Amount:** {waste_amount} kg ({waste_amount_tonnes:.3f} tonnes)")
                    st.write(f"**Process Conditions:** {TEMP}°C, {P} MPa, {RT} min")
                    st.write(f"**Composition:** C={C}%, H={H}%, N={N}%, O={O}%, SC={SC}%")
                    
            except Exception as e:
                st.error(f"❌ Prediction error: {str(e)}")

# Sidebar with information
with st.sidebar:
    st.header("ℹ️ About")
    st.markdown("""
    This app predicts hydrogen production using Supercritical Water Gasification (SCWG) technology.
    
    **Input Parameters:**
    - **Ultimate Analysis:** Chemical composition of waste
    - **Process Conditions:** SCWG operating parameters
    - **Waste Details:** Type and quantity
    
    **Output:**
    - Hydrogen yield and total production
    - Environmental impact metrics
    - Carbon reduction calculations
    """)
    
    st.header("📁 File Requirements")
    st.markdown("""
    Make sure these files are in the same directory:
    - `rf_model.pkl` - Trained Random Forest model
    - `scaler.pkl` - Feature scaler
    """)
    
    # Check if files exist
    if os.path.exists('rf_model.pkl'):
        st.success("✅ rf_model.pkl found")
    else:
        st.error("❌ rf_model.pkl missing")
    
    if os.path.exists('scaler.pkl'):
        st.success("✅ scaler.pkl found")
    else:
        st.error("❌ scaler.pkl missing")

# Footer
st.markdown("---")
st.markdown("*SCWG Hydrogen Production Predictor - Using Machine Learning for Sustainable Energy Solutions*")
