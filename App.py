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
    .hydrogen-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
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
    .validation-error {
        background-color: #ffe6e6;
        border-left: 4px solid #ff4d4d;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
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

TREE_SEQUESTRATION_FACTOR = 25  # kg CO2/tree/year
CAR_EMISSIONS_FACTOR = 0.250    # kg CO2/km for 1.8L gasoline car
BLUE_H2_SAVINGS_FACTOR = 1.6    # kg CO2 saved per kg H2 compared to gasoline

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
                                 help="Reaction temperature (300-650°C)")
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
                              help="Reaction pressure (10-35 MPa)")
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
        st.rerun()
    
    if st.button("Sludge Default", use_container_width=True):
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
    else:
        # Validate all inputs exactly as in your original code
        validation_errors = []
        
        # Validate Ultimate Analysis (C + H + N + O <= 100)
        ultimate_sum = C + H + N + O
        if ultimate_sum > 100:
            validation_errors.append("Ultimate Analysis has an issue (sum must be ≤100%)")
        
        # Validate Temperature range (300-650)
        if TEMP < 300 or TEMP > 650:
            validation_errors.append("Temperature is out of range (300-650°C)")
        
        # Validate Pressure range (10-35)
        if P < 10 or P > 35:
            validation_errors.append("Pressure is out of range (10-35 MPa)")
        
        # Validate Solid Content (SC < 100)
        if SC >= 100:
            validation_errors.append("Solid Content is out of range (must be <100%)")
        
        # Validate Waste amount (must be positive)
        if waste_amount <= 0:
            validation_errors.append("Waste amount must be greater than 0 kg")
        
        if validation_errors:
            for error in validation_errors:
                st.markdown(f'<div class="validation-error">❌ {error}</div>', unsafe_allow_html=True)
        else:
            try:
                # All validations passed, prepare features for prediction
                features = [C, H, N, O, SC, TEMP, P, RT]
                features_array = np.array(features).reshape(1, -1)
                
                # Scale the features using the loaded scaler
                features_scaled = scaler.transform(features_array)
                
                # Make prediction (H2 yield in mol/kg)
                with st.spinner('🔬 Analyzing parameters and predicting hydrogen yield...'):
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
                st.markdown("---")
                st.markdown('<h2 style="text-align: center; color: #1f77b4;">📊 Prediction Results</h2>', unsafe_allow_html=True)
                
                # Hydrogen Production Metrics
                st.subheader("🌱 Hydrogen Production")
                h2_col1, h2_col2, h2_col3 = st.columns(3)
                
                with h2_col1:
                    st.markdown('<div class="hydrogen-card">', unsafe_allow_html=True)
                    st.metric("H₂ Yield", f"{h2_yield:.2f} mol/kg", help="Hydrogen yield per kg of waste")
                    st.metric("Total H₂ Production", f"{total_h2_kg:.2f} kg", help="Total hydrogen produced in kilograms")
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with h2_col2:
                    st.markdown('<div class="prediction-card">', unsafe_allow_html=True)
                    st.metric("Total H₂ Moles", f"{total_h2_mol:.2f} mol", help="Total hydrogen produced in moles")
                    st.metric("Process Efficiency", f"{(h2_yield/30*100):.1f}%", 
                             delta="High" if h2_yield > 15 else "Medium", help="Efficiency compared to maximum expected yield")
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with h2_col3:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.metric("Waste Processed", f"{waste_amount:.2f} kg", help="Total waste material processed")
                    st.metric("Waste in Tonnes", f"{waste_amount_tonnes:.4f} tonnes", help="Waste amount converted to tonnes")
                    st.markdown('</div>', unsafe_allow_html=True)
                
                # Environmental Impact Metrics
                st.subheader("🌍 Environmental Impact")
                env_col1, env_col2, env_col3 = st.columns(3)
                
                with env_col1:
                    st.markdown('<div class="environment-card">', unsafe_allow_html=True)
                    st.metric("CO₂ Reduction", f"{carbon_reduction:.2f} kgCO₂e", 
                             help="Carbon dioxide equivalent reduced")
                    st.metric("Carbon Sequestration", f"{carbon_sequestration:.2f} tree-years", 
                             help="Equivalent tree sequestration")
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with env_col2:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.metric("Equivalent Car Travel", f"{car_travel_km:.2f} km", 
                             help="Equivalent car travel distance saved")
                    st.metric("CO₂ Saved vs Blue H₂", f"{co2_saved_h2:.2f} kgCO₂e", 
                             help="CO2 saved compared to blue hydrogen production")
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with env_col3:
                    st.markdown('<div class="prediction-card">', unsafe_allow_html=True)
                    st.metric("Waste Type", waste_type, help="Type of waste material processed")
                    st.metric("Carbon Reduction Factor", f"{CARBON_REDUCTION_FACTORS[waste_type]} kgCO₂e/tonne", 
                             help="Carbon reduction factor for selected waste type")
                    st.markdown('</div>', unsafe_allow_html=True)
                
                # Environmental Impact Chart
                st.subheader("📈 Environmental Impact Overview")
                impact_data = {
                    'Metric': ['CO₂ Reduction', 'Tree Sequestration', 'Car Travel', 'H₂ Production'],
                    'Value': [carbon_reduction, carbon_sequestration, car_travel_km, total_h2_kg],
                    'Unit': ['kgCO₂e', 'tree-years', 'km', 'kg H₂']
                }
                
                # Create a simple bar chart
                chart_dict = {
                    'CO₂ Reduction (kgCO₂e)': carbon_reduction,
                    'Tree Sequestration (tree-years)': carbon_sequestration,
                    'Car Travel Equivalent (km)': car_travel_km,
                    'H₂ Production (kg)': total_h2_kg
                }
                st.bar_chart(chart_dict)
                
                # Detailed results in expandable section
                with st.expander("📋 Detailed Analysis & Units", expanded=True):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("Input Parameters")
                        st.write(f"**Waste Type:** {waste_type}")
                        st.write(f"**Waste Amount:** {waste_amount:.2f} kg ({waste_amount_tonnes:.4f} tonnes)")
                        st.write(f"**Temperature:** {TEMP:.1f}°C")
                        st.write(f"**Pressure:** {P:.1f} MPa")
                        st.write(f"**Reaction Time:** {RT:.1f} min")
                        st.write(f"**Solid Content:** {SC:.2f}%")
                    
                    with col2:
                        st.subheader("Waste Composition")
                        st.write(f"**Carbon (C):** {C:.2f}%")
                        st.write(f"**Hydrogen (H):** {H:.2f}%")
                        st.write(f"**Nitrogen (N):** {N:.2f}%")
                        st.write(f"**Oxygen (O):** {O:.2f}%")
                        st.write(f"**Ultimate Analysis Sum:** {ultimate_sum:.2f}%")
                    
                    # Units information
                    st.subheader("📏 Measurement Units")
                    unit_col1, unit_col2, unit_col3 = st.columns(3)
                    
                    with unit_col1:
                        st.write("**Hydrogen Metrics:**")
                        st.write("- H₂ Yield: mol/kg")
                        st.write("- Total H₂: mol")
                        st.write("- Total H₂: kg")
                    
                    with unit_col2:
                        st.write("**Environmental Metrics:**")
                        st.write("- CO₂ Reduction: kgCO₂e")
                        st.write("- Carbon Sequestration: tree-years")
                        st.write("- Car Travel: km")
                    
                    with unit_col3:
                        st.write("**Conversion Factors:**")
                        st.write("- 1 mole H₂ = 0.002016 kg")
                        st.write("- Tree sequestration: 25 kgCO₂/tree/year")
                        st.write("- Car emissions: 0.250 kgCO₂/km")
                
                st.success("✅ Prediction completed successfully!")
                
            except Exception as e:
                st.error(f"❌ Prediction error: {str(e)}")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
<b>SCWG Hydrogen Production Predictor</b> - Using Machine Learning for Sustainable Energy Solutions 🌱
<br>
<small>All predictions are based on the trained machine learning model and conversion factors</small>
</div>
""", unsafe_allow_html=True)
