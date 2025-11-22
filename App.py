import streamlit as st
import pickle
import numpy as np
import os
from streamlit.components.v1 import html

# --- CONFIGURATION & INITIAL SETUP ---

# Set page configuration with a themed feel
st.set_page_config(
    page_title="SCWG Hydrogen Production Predictor",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a cleaner look and feel
st.markdown("""
<style>
/* Main container padding */
.main-content {
    padding-top: 2rem;
}

/* Sidebar styling */
.css-1lcbmhc, .css-1lcbmhc > div {
    background-color: #f0f2f6; /* Light gray background for sidebar */
}

/* Header style for results */
h3 {
    color: #007BFF; /* Primary color for section headers */
}

/* Metric card styling */
[data-testid="stMetric"] {
    background-color: #FFFFFF;
    border: 1px solid #E0E0E0;
    padding: 15px;
    border-radius: 10px;
    box-shadow: 2px 2px 5px rgba(0, 0, 0, 0.05);
}

/* Success/Error message styling */
.stSuccess, .stError {
    border-radius: 8px;
}

/* Submit button styling */
.stButton>button {
    width: 100%;
    border-radius: 8px;
    background-color: #28a745; /* Green color */
    color: white;
    font-weight: bold;
    height: 3rem;
    font-size: 1.1rem;
}
</style>
""", unsafe_allow_html=True)


# Load the trained model and scaler (using st.cache_data instead of st.cache_resource for objects)
@st.cache_data
def load_model():
    try:
        with open('rf_model.pkl', 'rb') as file:
            model = pickle.load(file)
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

@st.cache_data
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
    'Sewage Sludge': 310,       # Average of 150-550 kgCO2e/tonne
    'Lignocellulosic Biomass': 400, # Average of 150-450 kgCO2e/tonne
    'Petrochemical': 240         # Average of 400-900 kgCO2e/tonne
}

# Conversion factors
TREE_SEQUESTRATION_FACTOR = 25  # kg CO2/tree/year
CAR_EMISSIONS_FACTOR = 0.250    # kg CO2/km for 1.8L gasoline car
BLUE_H2_SAVINGS_FACTOR = 1.6    # kg CO2 saved per kg H2 compared to gasoline (simplified factor)

# --- MAIN APP LAYOUT ---

st.title("🧪 SCWG Hydrogen Production Predictor")
st.markdown("""
Predict **Hydrogen Production** from **Supercritical Water Gasification (SCWG)** and calculate the environmental benefits.
""")

# Use Tabs to separate input, prediction, and instructions
tab_predict, tab_info = st.tabs(["🚀 Predict H₂ Production", "💡 About & Instructions"])

with tab_predict:
    
    # Create input form with a sidebar for global controls
    with st.form("prediction_form_v2"):
        st.header("Input Parameters")
        
        # Two columns for inputs
        col_comp, col_proc = st.columns(2)
        
        with col_comp:
            st.subheader("🗑️ Waste Composition (Ultimate Analysis)")
            st.markdown("Enter the chemical composition percentages of the dry waste material.")
            
            # Group C and H, N and O for better visual flow
            comp_col1, comp_col2 = st.columns(2)
            
            with comp_col1:
                C = st.number_input("Carbon (C) %", min_value=0.0, max_value=100.0, value=50.0, step=0.1, help="Carbon percentage in the dry waste.")
                N = st.number_input("Nitrogen (N) %", min_value=0.0, max_value=100.0, value=2.0, step=0.1, help="Nitrogen percentage in the dry waste.")
            
            with comp_col2:
                H = st.number_input("Hydrogen (H) %", min_value=0.0, max_value=100.0, value=6.0, step=0.1, help="Hydrogen percentage in the dry waste.")
                O = st.number_input("Oxygen (O) %", min_value=0.0, max_value=100.0, value=30.0, step=0.1, help="Oxygen percentage in the dry waste.")
                
            # Calculate and display ultimate analysis sum
            ultimate_sum = C + H + N + O
            st.info(f"Ultimate Analysis Sum: **{ultimate_sum:.1f}%**")
            if ultimate_sum > 100:
                st.error("⚠️ Ultimate Analysis sum cannot exceed 100%")
        
        with col_proc:
            st.subheader("🌡️ Process Conditions & Feedstock")
            st.markdown("Specify the SCWG operating parameters and waste details.")
            
            # Process Condition Inputs
            SC = st.number_input("Solid Content (%)", min_value=0.1, max_value=99.9, value=15.0, step=0.1, help="Percentage of solid material in the aqueous feed slurry.")
            
            proc_col1, proc_col2 = st.columns(2)
            with proc_col1:
                TEMP = st.slider("Temperature (°C)", min_value=300, max_value=650, value=500, help="Reaction temperature (Supercritical range is > 374 °C).")
                P = st.slider("Pressure (MPa)", min_value=10, max_value=35, value=25, help="Reaction pressure (Critical pressure is 22.1 MPa).")
            with proc_col2:
                RT = st.number_input("Reaction Time (min)", min_value=0.0, value=30.0, step=1.0, help="Duration of the gasification process.")
                
            st.markdown("---")
            
            # Waste Details
            waste_col1, waste_col2 = st.columns(2)
            with waste_col1:
                waste_type = st.selectbox(
                    "Waste Type",
                    options=list(CARBON_REDUCTION_FACTORS.keys()),
                    index=0,
                    help="Select the waste category to estimate CO₂ reduction from avoiding land disposal."
                )
            with waste_col2:
                waste_amount = st.number_input("Waste Amount (kg)", min_value=0.1, value=100.0, step=1.0, help="Total mass of dry waste treated for this batch.")
        
        st.markdown("---")
        # Submit button centered in full width
        submitted = st.form_submit_button("🔥 Predict Hydrogen Production")
        st.markdown("---")

    # --- HANDLE PREDICTION & DISPLAY RESULTS ---
    if submitted:
        if model is None or scaler is None:
            st.error("❌ Model or scaler not loaded properly. Please check your .pkl files.")
        else:
            # Validate inputs
            if ultimate_sum > 100 or SC >= 100 or waste_amount <= 0:
                st.error("Please correct the input errors above before predicting.")
            else:
                try:
                    # Prepare features for prediction
                    features = [C, H, N, O, SC, TEMP, P, RT]
                    features_array = np.array(features).reshape(1, -1)
                    
                    # Scale the features
                    features_scaled = scaler.transform(features_array)
                    
                    # Make prediction
                    with st.spinner('Calculating H₂ yield and environmental impact...'):
                        h2_yield = model.predict(features_scaled)[0]
                    
                    # Post-prediction calculations
                    total_h2_mol = h2_yield * waste_amount
                    total_h2_kg = total_h2_mol * 0.002016  # 1 mole H2 = 0.002016 kg
                    
                    # Calculate CO2e reduction
                    waste_amount_tonnes = waste_amount / 1000
                    carbon_reduction = CARBON_REDUCTION_FACTORS.get(waste_type, 0) * waste_amount_tonnes
                    carbon_sequestration = carbon_reduction / TREE_SEQUESTRATION_FACTOR
                    car_travel_km = carbon_reduction / CAR_EMISSIONS_FACTOR
                    co2_saved_h2 = total_h2_kg * BLUE_H2_SAVINGS_FACTOR # CO2 saved by using Green H2 instead of Blue H2
                    
                    st.success("✅ Prediction completed successfully! Review the results below.")
                    st.subheader("📈 Predicted Results")
                    
                    # Results organized into two sections: H2 Production and Environmental Impact
                    
                    ### H2 Production Metrics ###
                    st.markdown("### Hydrogen Production")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("H₂ Yield", f"**{h2_yield:.2f}**", help="Moles of Hydrogen produced per kg of dry waste feed (mol/kg).")
                    
                    with col2:
                        st.metric("Total H₂ Production (kg)", f"**{total_h2_kg:.2f}**", help="Total mass of Hydrogen produced based on the input waste amount.")
                        
                    with col3:
                        st.metric("Total H₂ Production (mol)", f"**{total_h2_mol:.2f}**", help="Total moles of Hydrogen produced.")
                    
                    ### Environmental Impact Metrics ###
                    st.markdown("### Environmental Impact (Avoided Emissions)")
                    col4, col5, col6 = st.columns(3)
                    
                    with col4:
                        st.metric("CO₂ Reduction (SCWG vs Landfill)", f"**{carbon_reduction:.0f}** kg CO₂e", help=f"Estimated CO₂ equivalent saved by choosing SCWG over landfill for {waste_type}.")
                        
                    with col5:
                        st.metric("Equivalent Tree Sequestration", f"**{carbon_sequestration:.1f}** tree-years", help=f"Amount of CO₂ reduction equivalent to the sequestration by this many trees over one year (25 kg CO₂/tree/year).")
                    
                    with col6:
                        st.metric("Equivalent Car Travel Avoided", f"**{car_travel_km:.0f}** km", help=f"Amount of CO₂ reduction equivalent to the emissions from driving a standard car this far.")

                    # Use an expander for additional context
                    with st.expander("ℹ️ Detailed Context & Assumptions"):
                        st.markdown("#### Input Summary")
                        st.json({
                            "Waste Type": waste_type,
                            "Waste Amount": f"{waste_amount} kg",
                            "Composition": f"C:{C}%, H:{H}%, N:{N}%, O:{O}%",
                            "Process Conditions": f"SC:{SC}%, Temp:{TEMP}°C, Pres:{P} MPa, Time:{RT} min"
                        })
                        
                        st.markdown("#### Environmental Factor Breakdown")
                        st.table({
                            "Factor": ["CO₂ Reduction Factor", "Tree Sequestration", "Car Emissions"],
                            "Value": [f"{CARBON_REDUCTION_FACTORS.get(waste_type, 0)} kgCO₂e/tonne", f"{TREE_SEQUESTRATION_FACTOR} kg CO₂/tree/year", f"{CAR_EMISSIONS_FACTOR} kg CO₂/km"]
                        })
                        
                except Exception as e:
                    st.error(f"❌ An unexpected error occurred during prediction: {str(e)}")


with tab_info:
    st.header("💡 SCWG Explained")
    st.markdown("""
    **Supercritical Water Gasification (SCWG)** is a process that uses water above its critical point 
    ($374^{\circ}C$ and $22.1$ MPa) to convert wet biomass and organic waste into a hydrogen-rich gas.
    This method is highly efficient for wet feedstock as it bypasses energy-intensive drying processes.
    """)
    

with tab_info:
    st.header("💡 SCWG Explained")
    st.markdown("""
    **Supercritical Water Gasification (SCWG)** is a process that uses water above its critical point 
    ($374^{\circ}C$ and $22.1$ MPa) to convert wet biomass and organic waste into a hydrogen-rich gas.
    This method is highly efficient for wet feedstock as it bypasses energy-intensive drying processes.
    """)
    # st.image("path/to/your/image.png") # Use this if you have a local image file
    #  # <-- Removed the raw tag or commented it out

    st.header("📁 File Requirements & Status")
    st.markdown("""
    The application relies on two crucial files for prediction:
    1.  **`rf_model.pkl`**: The trained Machine Learning model (Random Forest Regressor).
    2.  **`scaler.pkl`**: The preprocessing scaler used on the training data.
    """)
    
    st.warning("⚠️ **IMPORTANT**: Both files must be present in the same directory as this Streamlit script.")
    
    # Check file status and display clearly
    col_model, col_scaler = st.columns(2)
    
    with col_model:
        if os.path.exists('rf_model.pkl'):
            st.success("✅ `rf_model.pkl` found and loaded.")
        else:
            st.error("❌ `rf_model.pkl` missing.")
    
    with col_scaler:
        if os.path.exists('scaler.pkl'):
            st.success("✅ `scaler.pkl` found and loaded.")
        else:
            st.error("❌ `scaler.pkl` missing.")
            
    st.header("⚙️ Data Inputs")
    st.markdown("""
    * **Ultimate Analysis (C, H, N, O)**: The chemical breakdown of the dry waste.
    * **Solid Content (SC)**: The concentration of the waste in the aqueous feed.
    * **Temperature (TEMP) & Pressure (P)**: The primary variables controlling the SCWG process efficiency.
    * **Reaction Time (RT)**: The duration for gasification to occur.
    """)


# --- FOOTER ---
st.markdown("---")
st.markdown("<p style='text-align: center; font-style: italic; color: grey;'>SCWG Hydrogen Production Predictor - Using Machine Learning for Sustainable Energy Solutions</p>", unsafe_allow_html=True)
