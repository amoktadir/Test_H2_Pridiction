import streamlit as st
import pickle
import numpy as np
import os

# Set page configuration
st.set_page_config(
    page_title="SCWG Hydrogen Production Predictor",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
    }
    .environment-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
    }
    .prediction-card {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
    }
    .input-section {
        background-color: #ffffff;
        padding: 2rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 2rem;
    }
    .number-input {
        background-color: #f8f9fa;
        border-radius: 5px;
        padding: 0.5rem;
        border: 1px solid #dee2e6;
    }
</style>
""", unsafe_allow_html=True)

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

# Constants
CARBON_REDUCTION_FACTORS = {
    'Sewage Sludge': 310,
    'Lignocellulosic Biomass': 400,
    'Petrochemical': 240
}

TREE_SEQUESTRATION_FACTOR = 25
CAR_EMISSIONS_FACTOR = 0.250
BLUE_H2_SAVINGS_FACTOR = 1.6

# Header Section
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown('<h1 class="main-header">⚡ SCWG Hydrogen Production Predictor</h1>', unsafe_allow_html=True)
    
st.markdown("""
<div style='text-align: center; font-size: 1.2rem; color: #666; margin-bottom: 3rem;'>
Predict hydrogen production from Supercritical Water Gasification (SCWG) of various waste materials
</div>
""", unsafe_allow_html=True)

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown('<div class="input-section">', unsafe_allow_html=True)
    
    # Input form with tabs for better organization
    tab1, tab2 = st.tabs(["🧪 Waste Composition", "⚙️ Process Conditions"])
    
    with tab1:
        st.subheader("Waste Composition Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="number-input">', unsafe_allow_html=True)
            C = st.number_input("Carbon (C) %", min_value=0.0, max_value=100.0, value=50.0, step=0.01, format="%.2f",
                               help="Carbon content in the waste material")
            H = st.number_input("Hydrogen (H) %", min_value=0.0, max_value=100.0, value=6.0, step=0.01, format="%.2f",
                               help="Hydrogen content in the waste material")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="number-input">', unsafe_allow_html=True)
            N = st.number_input("Nitrogen (N) %", min_value=0.0, max_value=100.0, value=2.0, step=0.01, format="%.2f",
                               help="Nitrogen content in the waste material")
            O = st.number_input("Oxygen (O) %", min_value=0.0, max_value=100.0, value=30.0, step=0.01, format="%.2f",
                               help="Oxygen content in the waste material")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Composition visualization using native Streamlit
        ultimate_sum = C + H + N + O
        remaining = max(0, 100 - ultimate_sum)
        
        st.subheader("Composition Breakdown")
        
        # Display composition as metrics
        comp_col1, comp_col2, comp_col3, comp_col4, comp_col5 = st.columns(5)
        
        with comp_col1:
            st.metric("Carbon", f"{C:.2f}%", delta=None, delta_color="off")
        with comp_col2:
            st.metric("Hydrogen", f"{H:.2f}%", delta=None, delta_color="off")
        with comp_col3:
            st.metric("Nitrogen", f"{N:.2f}%", delta=None, delta_color="off")
        with comp_col4:
            st.metric("Oxygen", f"{O:.2f}%", delta=None, delta_color="off")
        with comp_col5:
            st.metric("Other", f"{remaining:.2f}%", delta=None, delta_color="off")
        
        # Simple progress bar representation
        st.write("**Composition Visualization:**")
        total_width = 100
        composition_html = f"""
        <div style="display: flex; width: 100%; height: 30px; border-radius: 15px; overflow: hidden; margin: 10px 0;">
            <div style="background: #FF6B6B; width: {C}%; height: 100%;" title="Carbon: {C}%"></div>
            <div style="background: #4ECDC4; width: {H}%; height: 100%;" title="Hydrogen: {H}%"></div>
            <div style="background: #45B7D1; width: {N}%; height: 100%;" title="Nitrogen: {N}%"></div>
            <div style="background: #96CEB4; width: {O}%; height: 100%;" title="Oxygen: {O}%"></div>
            <div style="background: #FECA57; width: {remaining}%; height: 100%;" title="Other: {remaining}%"></div>
        </div>
        """
        st.markdown(composition_html, unsafe_allow_html=True)
        
        # Legend
        legend_col1, legend_col2, legend_col3, legend_col4, legend_col5 = st.columns(5)
        with legend_col1:
            st.markdown("🔴 **Carbon**")
        with legend_col2:
            st.markdown("🔵 **Hydrogen**")
        with legend_col3:
            st.markdown("🔷 **Nitrogen**")
        with legend_col4:
            st.markdown("💚 **Oxygen**")
        with legend_col5:
            st.markdown("💛 **Other**")
        
        if ultimate_sum > 100:
            st.error("⚠️ Ultimate Analysis sum cannot exceed 100%")
        else:
            st.success(f"✅ Composition sum: {ultimate_sum:.2f}% (Remaining: {remaining:.2f}%)")
    
    with tab2:
        st.subheader("Process Parameters")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="number-input">', unsafe_allow_html=True)
            SC = st.number_input("Solid Content (%)", min_value=0.1, max_value=99.9, value=15.0, step=0.01, format="%.2f",
                               help="Percentage of solid content in the waste")
            TEMP = st.number_input("Temperature (°C)", min_value=300.0, max_value=650.0, value=500.0, step=0.1, format="%.1f",
                                 help="Reaction temperature")
            waste_type = st.selectbox(
                "Waste Type",
                options=list(CARBON_REDUCTION_FACTORS.keys()),
                index=0,
                help="Type of waste material being processed"
            )
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="number-input">', unsafe_allow_html=True)
            P = st.number_input("Pressure (MPa)", min_value=10.0, max_value=35.0, value=25.0, step=0.1, format="%.1f",
                              help="Reaction pressure")
            RT = st.number_input("Reaction Time (min)", min_value=0.0, max_value=120.0, value=30.0, step=0.1, format="%.1f",
                               help="Duration of the reaction")
            waste_amount = st.number_input("Waste Amount (kg)", min_value=0.1, value=100.0, step=0.1, format="%.2f",
                                         help="Total amount of waste to be processed")
            st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    # Information panel
    st.markdown("""
    <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 2rem; border-radius: 10px; height: 100%;'>
    <h3 style='color: white; margin-bottom: 1rem;'>💡 About SCWG</h3>
    <p style='color: white; font-size: 0.9rem;'>
    Supercritical Water Gasification (SCWG) is an advanced technology that converts wet biomass and waste materials into hydrogen-rich syngas using water at supercritical conditions.
    </p>
    <h4 style='color: white; margin-top: 1.5rem;'>🎯 Key Benefits</h4>
    <ul style='color: white; font-size: 0.9rem;'>
    <li>High hydrogen yield</li>
    <li>Wet feedstock processing</li>
    <li>Reduced carbon emissions</li>
    <li>Waste-to-energy conversion</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Quick input presets
    st.markdown("### ⚡ Quick Presets")
    
    if st.button("Biomass Default", use_container_width=True):
        # Using session state to store values
        st.session_state.C = 50.0
        st.session_state.H = 6.0
        st.session_state.N = 2.0
        st.session_state.O = 30.0
        st.session_state.SC = 15.0
        st.rerun()
    
    if st.button("Sludge Default", use_container_width=True):
        st.session_state.C = 45.0
        st.session_state.H = 5.5
        st.session_state.N = 3.0
        st.session_state.O = 35.0
        st.session_state.SC = 12.5
        st.rerun()
    
    # File status
    st.markdown("### 📁 Model Status")
    status_col1, status_col2 = st.columns(2)
    with status_col1:
        if os.path.exists('rf_model.pkl'):
            st.success("✅ Model Ready")
        else:
            st.error("❌ Model Missing")
    with status_col2:
        if os.path.exists('scaler.pkl'):
            st.success("✅ Scaler Ready")
        else:
            st.error("❌ Scaler Missing")

# Initialize session state for inputs
if 'C' not in st.session_state:
    st.session_state.C = 50.0
if 'H' not in st.session_state:
    st.session_state.H = 6.0
if 'N' not in st.session_state:
    st.session_state.N = 2.0
if 'O' not in st.session_state:
    st.session_state.O = 30.0
if 'SC' not in st.session_state:
    st.session_state.SC = 15.0

# Prediction Button
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    predict_btn = st.button("🚀 Predict Hydrogen Production", 
                          use_container_width=True, 
                          type="primary",
                          disabled=(model is None or scaler is None))

# Handle prediction
if predict_btn:
    if model is None or scaler is None:
        st.error("❌ Model or scaler not loaded properly. Please check your .pkl files.")
    elif ultimate_sum > 100:
        st.error("⚠️ Please adjust the Ultimate Analysis values (sum must be ≤100%)")
    elif SC >= 100:
        st.error("⚠️ Solid Content must be less than 100%")
    elif waste_amount <= 0:
        st.error("⚠️ Waste amount must be greater than 0 kg")
    else:
        try:
            # Prepare features and make prediction
            features = [C, H, N, O, SC, TEMP, P, RT]
            features_array = np.array(features).reshape(1, -1)
            features_scaled = scaler.transform(features_array)
            
            with st.spinner('🔬 Analyzing parameters and predicting hydrogen yield...'):
                h2_yield = model.predict(features_scaled)[0]
            
            # Calculate results
            total_h2_mol = h2_yield * waste_amount
            total_h2_kg = total_h2_mol * 0.002016
            waste_amount_tonnes = waste_amount / 1000
            carbon_reduction = CARBON_REDUCTION_FACTORS.get(waste_type, 0) * waste_amount_tonnes
            carbon_sequestration = carbon_reduction / TREE_SEQUESTRATION_FACTOR
            car_travel_km = carbon_reduction / CAR_EMISSIONS_FACTOR
            co2_saved_h2 = total_h2_kg * BLUE_H2_SAVINGS_FACTOR
            
            # Results Section
            st.markdown("---")
            st.markdown('<h2 style="text-align: center; color: #1f77b4;">📊 Prediction Results</h2>', unsafe_allow_html=True)
            
            # Main metrics in cards
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown('<div class="prediction-card">', unsafe_allow_html=True)
                st.metric("H₂ Yield", f"{h2_yield:.3f} mol/kg", delta="Optimal" if h2_yield > 10 else "Good")
                st.metric("Total H₂ Production", f"{total_h2_kg:.3f} kg")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown('<div class="environment-card">', unsafe_allow_html=True)
                st.metric("CO₂ Reduction", f"{carbon_reduction:.2f} kgCO₂e", 
                         delta=f"Equivalent to {carbon_sequestration:.1f} trees")
                st.metric("Car Travel Equivalent", f"{car_travel_km:.1f} km")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col3:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("Process Efficiency", f"{(h2_yield/30*100):.2f}%", 
                         delta="High" if h2_yield > 15 else "Medium")
                st.metric("CO₂ Saved vs Blue H₂", f"{co2_saved_h2:.2f} kgCO₂e")
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Environmental Impact using native Streamlit chart
            st.subheader("🌍 Environmental Impact Overview")
            
            # Create a simple bar chart data
            impact_df = {
                'Metrics': ['CO₂ Reduction', 'Tree Years', 'Car Travel', 'H₂ Production'],
                'Values': [carbon_reduction, carbon_sequestration, car_travel_km, total_h2_kg]
            }
            
            # Display as bar chart
            st.bar_chart(data=impact_df, x='Metrics', y='Values', use_container_width=True)
            
            # Detailed results in expandable section
            with st.expander("📋 Detailed Analysis", expanded=True):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Input Summary")
                    st.write(f"**Waste Type:** {waste_type}")
                    st.write(f"**Waste Amount:** {waste_amount:.2f} kg ({waste_amount_tonnes:.4f} tonnes)")
                    st.write(f"**Temperature:** {TEMP:.1f}°C")
                    st.write(f"**Pressure:** {P:.1f} MPa")
                    st.write(f"**Reaction Time:** {RT:.1f} min")
                
                with col2:
                    st.subheader("Composition")
                    st.write(f"**Carbon (C):** {C:.2f}%")
                    st.write(f"**Hydrogen (H):** {H:.2f}%")
                    st.write(f"**Nitrogen (N):** {N:.2f}%")
                    st.write(f"**Oxygen (O):** {O:.2f}%")
                    st.write(f"**Solid Content:** {SC:.2f}%")
            
        except Exception as e:
            st.error(f"❌ Prediction error: {str(e)}")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
<b>SCWG Hydrogen Production Predictor</b> - Using Machine Learning for Sustainable Energy Solutions 🌱
</div>
""", unsafe_allow_html=True)
