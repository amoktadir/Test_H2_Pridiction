import streamlit as st
import pickle
import numpy as np
import pandas as pd
import os
import subprocess
import sys

# Set page configuration
st.set_page_config(
    page_title="H₂ Yield Predictor",
    page_icon="⚗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

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

def install_required_packages():
    """Install required packages if missing"""
    try:
        import sklearn
    except ImportError:
        st.warning("Installing required packages...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "scikit-learn==1.3.0"])
        st.success("Packages installed successfully!")
        st.experimental_rerun()

def load_model():
    """Load the trained model with error handling"""
    try:
        with open('rf_model.pkl', 'rb') as file:
            model = pickle.load(file)
        return model
    except FileNotFoundError:
        st.error("Error: Model file 'rf_model.pkl' not found.")
        return None
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def load_scaler():
    """Load the scaler with error handling"""
    try:
        with open('scaler.pkl', 'rb') as file:
            scaler = pickle.load(file)
        return scaler
    except FileNotFoundError:
        st.error("Error: Scaler file 'scaler.pkl' not found.")
        return None
    except Exception as e:
        st.error(f"Error loading scaler: {str(e)}")
        return None

def main():
    # Custom CSS for better styling
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2e86ab;
        margin-bottom: 1rem;
    }
    .result-box {
        background-color: #f0f8ff;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 1rem 0;
    }
    .metric-box {
        background-color: #e6f3ff;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }
    </style>
    """, unsafe_allow_html=True)

    # Header
    st.markdown('<h1 class="main-header">⚗️ Hydrogen Yield Predictor</h1>', unsafe_allow_html=True)
    st.markdown("Predict H₂ yield from Supercritical Water Gasification (SCWG) process")

    # Install required packages
    install_required_packages()

    # Load model and scaler
    model = load_model()
    scaler = load_scaler()

    if model is None or scaler is None:
        st.error("""
        **Model files not found or incompatible!**
        
        Please run the model regeneration script first:
        ```bash
        python regenerate_model.py
        ```
        """)
        return

    # Create two columns for layout
    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown('<h3 class="sub-header">📊 Process Parameters</h3>', unsafe_allow_html=True)
        
        # Input form
        with st.form("prediction_form"):
            st.markdown("**Feedstock Composition (wt%)**")
            
            # Create columns for Ultimate Analysis - ALL VALUES AS FLOATS
            col1a, col1b, col1c, col1d = st.columns(4)
            with col1a:
                C = st.number_input("Carbon (C)", min_value=0.0, max_value=100.0, value=48.9, step=0.1, help="Carbon content in wt%")
            with col1b:
                H = st.number_input("Hydrogen (H)", min_value=0.0, max_value=100.0, value=6.2, step=0.1, help="Hydrogen content in wt%")
            with col1c:
                N = st.number_input("Nitrogen (N)", min_value=0.0, max_value=100.0, value=1.0, step=0.1, help="Nitrogen content in wt%")
            with col1d:
                O = st.number_input("Oxygen (O)", min_value=0.0, max_value=100.0, value=40.6, step=0.1, help="Oxygen content in wt%")
            
            st.markdown("**Process Conditions**")
            col2a, col2b, col2c, col2d = st.columns(4)
            with col2a:
                SC = st.number_input("Steam-to-Carbon (SC)", min_value=0.0, max_value=50.0, value=10.0, step=0.1, help="Steam-to-carbon ratio")
            with col2b:
                # FIXED: All values as floats to avoid mixed types
                TEMP = st.number_input("Temperature (°C)", min_value=300.0, max_value=650.0, value=600.0, step=1.0, help="Reactor temperature (300-650°C)")
            with col2c:
                # FIXED: All values as floats to avoid mixed types
                P = st.number_input("Pressure (MPa)", min_value=10.0, max_value=35.0, value=25.0, step=1.0, help="Reactor pressure (10-35 MPa)")
            with col2d:
                RT = st.number_input("Residence Time (min)", min_value=0.0, max_value=200.0, value=50.0, step=1.0, help="Residence time in minutes")
            
            st.markdown("**Waste Input**")
            col3a, col3b = st.columns(2)
            with col3a:
                waste_amount = st.number_input("Waste Amount (kg)", min_value=0.1, value=100.0, step=10.0, help="Amount of waste to process")
            with col3b:
                waste_type = st.selectbox(
                    "Waste Type",
                    options=list(CARBON_REDUCTION_FACTORS.keys()),
                    index=0,
                    help="Type of waste material"
                )
            
            # Submit button
            submitted = st.form_submit_button("🚀 Predict H₂ Yield", use_container_width=True)

    with col2:
        st.markdown('<h3 class="sub-header">📈 Prediction Results</h3>', unsafe_allow_html=True)
        
        if submitted:
            try:
                # Validate Ultimate Analysis (C + H + N + O <= 100)
                ultimate_sum = C + H + N + O
                if ultimate_sum > 100:
                    st.error(f"❌ Ultimate Analysis sum ({ultimate_sum:.1f}%) exceeds 100%. Please adjust the composition.")
                    return
                
                # Validate Temperature range (300-650)
                if TEMP < 300.0 or TEMP > 650.0:
                    st.error("❌ Temperature is out of range (300-650°C)")
                    return
                
                # Validate Pressure range (10-35)
                if P < 10.0 or P > 35.0:
                    st.error("❌ Pressure is out of range (10-35 MPa)")
                    return
                
                # Validate Solid Content (SC < 100)
                if SC >= 100.0:
                    st.error("❌ SC is out of range (must be <100%)")
                    return
                
                # Validate Waste amount (must be positive)
                if waste_amount <= 0:
                    st.error("❌ Waste amount must be greater than 0 kg")
                    return
                
                # All validations passed, prepare features for prediction
                features = [C, H, N, O, SC, TEMP, P, RT]
                
                # Convert to numpy array and reshape for prediction
                features_array = np.array(features).reshape(1, -1)
                
                # Scale the features using the loaded scaler
                features_scaled = scaler.transform(features_array)
                
                # Make prediction (H2 yield in mol/kg)
                h2_yield = model.predict(features_scaled)[0]
                
                # Calculate total hydrogen production (mol)
                total_h2_mol = h2_yield * waste_amount
                
                # Convert total hydrogen production to kilograms
                total_h2_kg = total_h2_mol * 0.002016  # 1 mole H2 = 0.002016 kg
                
                # Calculate CO2e reduction (convert waste amount from kg to tonnes)
                waste_amount_tonnes = waste_amount / 1000
                carbon_reduction = CARBON_REDUCTION_FACTORS.get(waste_type, 0) * waste_amount_tonnes
                
                # Calculate carbon sequestration in tree-years
                carbon_sequestration = carbon_reduction / TREE_SEQUESTRATION_FACTOR
                
                # Calculate equivalent car travel distance
                car_travel_km = carbon_reduction / CAR_EMISSIONS_FACTOR
                
                # Calculate CO2 saved from blue H2 vs gasoline
                co2_saved_h2 = total_h2_kg * BLUE_H2_SAVINGS_FACTOR
                
                # Display results
                st.markdown('<div class="result-box">', unsafe_allow_html=True)
                
                # Hydrogen Production Metrics
                st.markdown("### 🎯 Hydrogen Production")
                col_res1, col_res2, col_res3 = st.columns(3)
                with col_res1:
                    st.metric("H₂ Yield", f"{h2_yield:.2f} mol/kg")
                with col_res2:
                    st.metric("Total H₂ (moles)", f"{total_h2_mol:.2f} mol")
                with col_res3:
                    st.metric("Total H₂ (mass)", f"{total_h2_kg:.2f} kg")
                
                # Environmental Impact Metrics
                st.markdown("### 🌱 Environmental Impact")
                col_env1, col_env2, col_env3 = st.columns(3)
                with col_env1:
                    st.metric("CO₂ Reduction", f"{carbon_reduction:.2f} kgCO₂e")
                with col_env2:
                    st.metric("Equivalent Trees", f"{carbon_sequestration:.1f} trees/year")
                with col_env3:
                    st.metric("Car Travel Saved", f"{car_travel_km:.0f} km")
                
                st.markdown("### 💧 Blue H₂ Comparison")
                st.metric("CO₂ Saved vs Gasoline", f"{co2_saved_h2:.2f} kgCO₂e")
                
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Additional information
                with st.expander("📋 Process Summary"):
                    st.write(f"""
                    **Feedstock Composition:**
                    - Carbon (C): {C} wt%
                    - Hydrogen (H): {H} wt%
                    - Nitrogen (N): {N} wt%
                    - Oxygen (O): {O} wt%
                    
                    **Process Conditions:**
                    - Temperature: {TEMP}°C
                    - Pressure: {P} MPa
                    - Steam-to-Carbon Ratio: {SC}
                    - Residence Time: {RT} min
                    
                    **Waste Input:**
                    - Type: {waste_type}
                    - Amount: {waste_amount} kg
                    """)
                
            except Exception as e:
                st.error(f"❌ An error occurred during prediction: {str(e)}")
        else:
            st.info("👆 Fill in the form and click 'Predict H₂ Yield' to see results")
            
            # Display sample data ranges
            with st.expander("📊 Typical Input Ranges"):
                st.write("""
                **Typical Input Ranges:**
                - Temperature: 300-650°C
                - Pressure: 10-35 MPa
                - Steam-to-Carbon: 0.5-30
                - Residence Time: 5-180 min
                - Ultimate Analysis: Sum should be ≤100%
                """)

    # Footer
    st.markdown("---")
    st.markdown(
        "**About**: This app predicts hydrogen yield from Supercritical Water Gasification using machine learning. "
        "The model is based on experimental data and considers various process parameters."
    )

if __name__ == '__main__':
    main()
