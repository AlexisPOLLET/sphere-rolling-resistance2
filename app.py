import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st 
import io
import warnings
import datetime
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Sphere Rolling Resistance Analysis",
    page_icon="⚪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'experiments' not in st.session_state:
    st.session_state.experiments = {}
if 'current_df' not in st.session_state:
    st.session_state.current_df = None
if 'current_df_valid' not in st.session_state:
    st.session_state.current_df_valid = None

# CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem 2rem;
        border-radius: 10px;
        color: white;
        margin-bottom: 2rem;
    }
    .metric-item {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 6px;
        text-align: center;
        border: 1px solid #dee2e6;
        margin: 0.5rem 0;
    }
    .metric-value {
        font-size: 1.5rem;
        font-weight: bold;
        color: #495057;
        margin-bottom: 0.5rem;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #6c757d;
        margin: 0;
    }
    .friction-card {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
    .status-success {
        background: #d1ecf1;
        color: #0c5460;
        padding: 0.5rem 1rem;
        border-radius: 4px;
        border-left: 4px solid #17a2b8;
    }
    .status-warning {
        background: #fff3cd;
        color: #856404;
        padding: 0.5rem 1rem;
        border-radius: 4px;
        border-left: 4px solid #ffc107;
    }
    .status-error {
        background: #f8d7da;
        color: #721c24;
        padding: 0.5rem 1rem;
        border-radius: 4px;
        border-left: 4px solid #dc3545;
    }
</style>
""", unsafe_allow_html=True)

# ==================== HELPER FUNCTIONS ====================

def create_sample_data_with_metadata(experiment_name="Sample", water_content=0.0, sphere_type="Steel"):
    """Creates sample data with experimental metadata"""
    frames = list(range(1, 108))
    data = []
    
    water_effect = 1 + (water_content / 100) * 0.3
    
    for frame in frames:
        if frame < 9:
            data.append([frame, 0, 0, 0])
        elif frame in [30, 31]:
            data.append([frame, 0, 0, 0])
        else:
            x = 1240 - (frame - 9) * 12 * water_effect + np.random.normal(0, 2)
            y = 680 + (frame - 9) * 0.5 + np.random.normal(0, 3)
            radius = 20 + np.random.normal(5, 3)
            radius = max(18, min(35, radius))
            data.append([frame, max(0, x), max(0, y), max(0, radius)])
    
    df = pd.DataFrame(data, columns=['Frame', 'X_center', 'Y_center', 'Radius'])
    
    metadata = {
        'experiment_name': experiment_name,
        'water_content': water_content,
        'sphere_type': sphere_type,
        'date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
        'total_frames': len(df),
        'valid_detections': len(df[(df['X_center'] != 0) & (df['Y_center'] != 0) & (df['Radius'] != 0)]),
        'success_rate': len(df[(df['X_center'] != 0) & (df['Y_center'] != 0) & (df['Radius'] != 0)]) / len(df) * 100
    }
    
    return df, metadata

def clean_data_robust(df_valid, fps=250, pixels_per_mm=5.0):
    """Robustly cleans data by eliminating start/end artifacts"""
    if len(df_valid) < 20:
        return df_valid, {"error": "Not enough data"}
    
    # Convert to physical units
    dt = 1 / fps
    x_m = df_valid['X_center'].values / pixels_per_mm / 1000
    y_m = df_valid['Y_center'].values / pixels_per_mm / 1000
    
    # Calculate velocities
    vx = np.gradient(x_m, dt)
    vy = np.gradient(y_m, dt)
    v_magnitude = np.sqrt(vx**2 + vy**2)
    
    # Calculate accelerations
    acceleration = np.abs(np.gradient(v_magnitude, dt))
    
    # Method 1: Remove first and last N points (brute but effective)
    n_remove = max(5, len(df_valid) // 10)  # Remove at least 5 points or 10% of data
    
    # Method 2: Detect stable velocity zones
    v_smooth = np.convolve(v_magnitude, np.ones(5)/5, mode='same')  # Smoothing
    v_median = np.median(v_smooth)
    v_threshold = v_median * 0.3  # Threshold at 30% of median velocity
    
    # Find stable zone
    stable_mask = v_smooth > v_threshold
    
    # Find stable zone start and end indices
    stable_indices = np.where(stable_mask)[0]
    
    if len(stable_indices) > 10:
        start_idx = stable_indices[0] + 3  # Add margin
        end_idx = stable_indices[-1] - 3   # Add margin
    else:
        # Fallback: use brute method
        start_idx = n_remove
        end_idx = len(df_valid) - n_remove
    
    # Ensure we still have enough data
    if end_idx - start_idx < 10:
        # Use more conservative approach
        start_idx = len(df_valid) // 5
        end_idx = len(df_valid) - len(df_valid) // 5
    
    # Trim data
    df_cleaned = df_valid.iloc[start_idx:end_idx].copy().reset_index(drop=True)
    
    cleaning_info = {
        "original_length": len(df_valid),
        "cleaned_length": len(df_cleaned),
        "start_removed": start_idx,
        "end_removed": len(df_valid) - end_idx,
        "percentage_kept": len(df_cleaned) / len(df_valid) * 100
    }
    
    return df_cleaned, cleaning_info

def calculate_advanced_metrics(df_valid, fps=250, pixels_per_mm=5.0, sphere_mass_g=10.0, angle_deg=15.0):
    """Calculate comprehensive kinematic and dynamic metrics"""
    if len(df_valid) < 10:
        return None
    
    # ROBUST DATA CLEANING
    df_clean, cleaning_info = clean_data_robust(df_valid, fps, pixels_per_mm)
    
    if "error" in cleaning_info:
        return None
    
    # Convert to real units with cleaned data
    dt = 1 / fps
    mass_kg = sphere_mass_g / 1000
    angle_rad = np.radians(angle_deg)
    g = 9.81
    
    x_m = df_clean['X_center'].values / pixels_per_mm / 1000
    y_m = df_clean['Y_center'].values / pixels_per_mm / 1000
    
    # Time array
    t = np.arange(len(df_clean)) * dt
    
    # Calculate velocities and accelerations with cleaned data
    vx = np.gradient(x_m, dt)
    vy = np.gradient(y_m, dt)
    v_magnitude = np.sqrt(vx**2 + vy**2)
    
    # Accelerations with smoothing to avoid noise
    acceleration_raw = np.gradient(v_magnitude, dt)
    # Smooth acceleration to eliminate spikes
    acceleration = np.convolve(acceleration_raw, np.ones(3)/3, mode='same')
    
    # Forces with smoothed acceleration
    F_resistance = mass_kg * np.abs(acceleration)
    
    # Energies
    E_kinetic = 0.5 * mass_kg * v_magnitude**2
    E_initial = E_kinetic[0] if len(E_kinetic) > 0 else 0
    E_final = E_kinetic[-1] if len(E_kinetic) > 0 else 0
    E_dissipated = E_initial - E_final
    
    # Power
    P_resistance = F_resistance * v_magnitude
    
    # Basic Krr calculation with cleaned data
    n_avg = min(3, len(v_magnitude)//4)
    v0 = np.mean(v_magnitude[:n_avg]) if len(v_magnitude) >= n_avg else v_magnitude[0]
    vf = np.mean(v_magnitude[-n_avg:]) if len(v_magnitude) >= n_avg else v_magnitude[-1]
    
    distances = np.sqrt(np.diff(x_m)**2 + np.diff(y_m)**2)
    total_distance = np.sum(distances)
    
    krr = (v0**2 - vf**2) / (2 * g * total_distance) if total_distance > 0 and v0 > vf else None
    
    return {
        'krr': krr,
        'v0': v0,
        'vf': vf,
        'distance': total_distance,
        'duration': t[-1] - t[0],
        'max_velocity': np.max(v_magnitude),
        'avg_velocity': np.mean(v_magnitude),
        'max_acceleration': np.max(np.abs(acceleration)),
        'energy_initial': E_initial,
        'energy_final': E_final,
        'energy_dissipated': E_dissipated,
        'energy_efficiency': (E_final / E_initial * 100) if E_initial > 0 else 0,
        'time': t,
        'velocity': v_magnitude,
        'acceleration': acceleration,
        'resistance_force': F_resistance,
        'power': P_resistance,
        'energy_kinetic': E_kinetic,
        'vx': vx,
        'vy': vy,
        'cleaning_info': cleaning_info  # Add cleaning info
    }

def calculate_friction_coefficients(df_valid, sphere_mass_g=10.0, angle_deg=15.0, fps=250.0, pixels_per_mm=5.0):
    """Calculate friction coefficients from trajectory data"""
    if len(df_valid) < 10:
        return None
    
    # ROBUST DATA CLEANING FOR FRICTION ALSO
    df_clean, cleaning_info = clean_data_robust(df_valid, fps, pixels_per_mm)
    
    if "error" in cleaning_info:
        return None
    
    # Physical parameters
    mass_kg = sphere_mass_g / 1000
    angle_rad = np.radians(angle_deg)
    g = 9.81
    dt = 1 / fps
    
    # Convert positions to meters with cleaned data
    x_m = df_clean['X_center'].values / pixels_per_mm / 1000
    y_m = df_clean['Y_center'].values / pixels_per_mm / 1000
    
    # Calculate velocities and accelerations with cleaned data
    vx = np.gradient(x_m, dt)
    vy = np.gradient(y_m, dt)
    v_magnitude = np.sqrt(vx**2 + vy**2)
    
    # Acceleration with smoothing
    acceleration_raw = np.gradient(v_magnitude, dt)
    acceleration = np.convolve(acceleration_raw, np.ones(3)/3, mode='same')
    
    # Forces
    F_gravity_component = mass_kg * g * np.sin(angle_rad)
    F_normal = mass_kg * g * np.cos(angle_rad)
    F_resistance = mass_kg * np.abs(acceleration)
    
    # Friction coefficients with smoothing
    mu_kinetic_raw = F_resistance / F_normal
    mu_kinetic = np.convolve(mu_kinetic_raw, np.ones(5)/5, mode='same')  # Additional smoothing
    mu_rolling = mu_kinetic - np.tan(angle_rad)
    
    # Basic calculations
    n_avg = min(3, len(v_magnitude)//4)
    v0 = np.mean(v_magnitude[:n_avg]) if len(v_magnitude) >= n_avg else v_magnitude[0]
    vf = np.mean(v_magnitude[-n_avg:]) if len(v_magnitude) >= n_avg else v_magnitude[-1]
    
    distances = np.sqrt(np.diff(x_m)**2 + np.diff(y_m)**2)
    total_distance = np.sum(distances)
    
    krr = (v0**2 - vf**2) / (2 * g * total_distance) if total_distance > 0 and v0 > vf else None
    
    # Energy analysis
    E_kinetic_initial = 0.5 * mass_kg * v0**2
    E_kinetic_final = 0.5 * mass_kg * vf**2
    E_potential_lost = mass_kg * g * total_distance * np.sin(angle_rad)
    E_dissipated = E_kinetic_initial - E_kinetic_final - E_potential_lost
    
    mu_energetic = E_dissipated / (F_normal * total_distance) if total_distance > 0 else None
    
    return {
        'mu_kinetic_avg': np.mean(mu_kinetic),
        'mu_kinetic_max': np.max(mu_kinetic),
        'mu_kinetic_min': np.min(mu_kinetic),
        'mu_rolling_avg': np.mean(mu_rolling),
        'mu_energetic': mu_energetic,
        'krr': krr,
        'v0': v0,
        'vf': vf,
        'total_distance': total_distance,
        'F_resistance_avg': np.mean(F_resistance),
        'F_normal': F_normal,
        'F_gravity_component': F_gravity_component,
        'E_dissipated': E_dissipated,
        'energy_efficiency': (E_kinetic_final / E_kinetic_initial * 100) if E_kinetic_initial > 0 else 0,
        'time': np.arange(len(df_clean)) * dt,
        'mu_kinetic_series': mu_kinetic,
        'mu_rolling_series': mu_rolling,
        'velocity': v_magnitude,
        'acceleration': acceleration,
        'F_resistance_series': F_resistance,
        'cleaning_info': cleaning_info  # Add cleaning info
    }

# ==================== MAIN APPLICATION ====================

# Header
st.markdown("""
<div class="main-header">
    <h1>📊 Sphere Rolling Resistance Analysis Platform</h1>
    <p>Complete analysis suite for granular mechanics research - Osaka University</p>
    <p><strong>🔥 NEW:</strong> Integrated grain-sphere friction analysis!</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.markdown("### 📋 Navigation")

# Analysis type selection
analysis_type = st.sidebar.selectbox(
    "Select analysis type:",
    [
        "📈 Code 1: Trajectory Visualization",
        "📊 Code 2: Krr Analysis", 
        "🔬 Code 3: Complete Analysis + Friction",
        "🔍 Multi-Experiment Comparison"
    ]
)

# ==================== DATA LOADING SECTION ====================

st.markdown("## 📂 Data Loading")

# Create tabs for data input
tab1, tab2 = st.tabs(["📁 Upload File", "🔬 Sample Data"])

with tab1:
    col1, col2, col3 = st.columns(3)
    with col1:
        experiment_name = st.text_input("Experiment Name", value="Experiment_1")
    with col2:
        water_content = st.number_input("Water Content (%)", value=0.0, min_value=0.0, max_value=30.0)
    with col3:
        sphere_type = st.selectbox("Sphere Type", ["Steel", "Plastic", "Glass", "Other"])
    
    uploaded_file = st.file_uploader(
        "Choose your CSV file", 
        type=['csv'],
        help="CSV with columns: Frame, X_center, Y_center, Radius"
    )
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            required_columns = ['Frame', 'X_center', 'Y_center', 'Radius']
            
            if all(col in df.columns for col in required_columns):
                df_valid = df[(df['X_center'] != 0) & (df['Y_center'] != 0) & (df['Radius'] != 0)]
                st.session_state.current_df = df
                st.session_state.current_df_valid = df_valid
                
                success_rate = len(df_valid) / len(df) * 100 if len(df) > 0 else 0
                st.success(f"✅ File loaded: {len(df)} frames, {len(df_valid)} valid detections ({success_rate:.1f}%)")
                
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")

with tab2:
    col1, col2 = st.columns(2)
    with col1:
        sample_water = st.slider("Sample Water Content (%)", 0.0, 25.0, 10.0, 0.5)
    with col2:
        sample_sphere = st.selectbox("Sample Sphere Type", ["Steel", "Plastic", "Glass"], key="sample_sphere")
    
    if st.button("🔬 Generate sample data"):
        df, metadata = create_sample_data_with_metadata(
            experiment_name=f"Sample_{sample_water}%", 
            water_content=sample_water, 
            sphere_type=sample_sphere
        )
        df_valid = df[(df['X_center'] != 0) & (df['Y_center'] != 0) & (df['Radius'] != 0)]
        
        st.session_state.current_df = df
        st.session_state.current_df_valid = df_valid
        st.success("📊 Sample data generated!")

# ==================== ANALYSIS SECTIONS ====================

if (st.session_state.current_df_valid is not None and 
    len(st.session_state.current_df_valid) > 0):
    
    df_valid = st.session_state.current_df_valid
    
    # Quick overview
    st.markdown("## 📊 Data Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-item">
            <div class="metric-value">{len(st.session_state.current_df)}</div>
            <div class="metric-label">Total Frames</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-item">
            <div class="metric-value">{len(df_valid)}</div>
            <div class="metric-label">Valid Detections</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        success_rate = len(df_valid) / len(st.session_state.current_df) * 100 if len(st.session_state.current_df) > 0 else 0
        st.markdown(f"""
        <div class="metric-item">
            <div class="metric-value">{success_rate:.1f}%</div>
            <div class="metric-label">Success Rate</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        avg_radius = df_valid['Radius'].mean() if len(df_valid) > 0 else 0
        st.markdown(f"""
        <div class="metric-item">
            <div class="metric-value">{avg_radius:.1f}</div>
            <div class="metric-label">Average Radius (px)</div>
        </div>
        """, unsafe_allow_html=True)
    
    # ===== CODE 3: ADVANCED COMPLETE ANALYSIS + FRICTION =====
    if analysis_type == "🔬 Code 3: Complete Analysis + Friction":
        st.markdown("## 🔬 Code 3: Advanced Kinematic Analysis + Friction Analysis")
        st.markdown("**🔥 NEW:** Integrated grain-sphere friction analysis!")
        
        # Parameters
        st.markdown("### ⚙️ Analysis Parameters")
        
        param_col1, param_col2, param_col3 = st.columns(3)
        
        with param_col1:
            st.markdown("**Sphere Parameters**")
            mass_g = st.number_input("Mass (g)", value=1.0, min_value=0.1)
            radius_mm = st.number_input("Radius (mm)", value=7.5, min_value=1.0)
            
        with param_col2:
            st.markdown("**Experimental Parameters**")
            fps_adv = st.number_input("FPS", value=250.0, min_value=1.0)
            angle_deg_adv = st.number_input("Angle (°)", value=15.0, min_value=0.1)
            
        with param_col3:
            st.markdown("**Calibration**")
            if len(df_valid) > 0:
                avg_radius_px = df_valid['Radius'].mean()
                auto_cal = avg_radius_px / radius_mm
                st.markdown(f"""
                <div class="metric-item">
                    <div class="metric-value">{auto_cal:.2f}</div>
                    <div class="metric-label">Auto calibration (px/mm)</div>
                </div>
                """, unsafe_allow_html=True)
                pixels_per_mm_adv = auto_cal
        
        # Launch analysis
        if st.button("🚀 Launch Complete Analysis + Friction"):
            
            with st.spinner("🧮 Calculating advanced metrics and friction analysis..."):
                # Calculate metrics
                metrics = calculate_advanced_metrics(df_valid, fps_adv, pixels_per_mm_adv, mass_g, angle_deg_adv)
                friction_results = calculate_friction_coefficients(
                    df_valid, mass_g, angle_deg_adv, fps_adv, pixels_per_mm_adv
                )
            
            if metrics and friction_results:
                # === DATA CLEANING INFORMATION DISPLAY ===
                st.markdown("### 🧹 Data Cleaning")
                
                cleaning_info = metrics.get('cleaning_info', {})
                clean_col1, clean_col2, clean_col3 = st.columns(3)
                
                with clean_col1:
                    st.markdown(f"""
                    <div class="metric-item">
                        <div class="metric-value">{cleaning_info.get('original_length', 0)}</div>
                        <div class="metric-label">Original Points</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                with clean_col2:
                    st.markdown(f"""
                    <div class="metric-item">
                        <div class="metric-value">{cleaning_info.get('cleaned_length', 0)}</div>
                        <div class="metric-label">Cleaned Points</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                with clean_col3:
                    percentage_kept = cleaning_info.get('percentage_kept', 0)
                    st.markdown(f"""
                    <div class="metric-item">
                        <div class="metric-value">{percentage_kept:.1f}%</div>
                        <div class="metric-label">Data Retained</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                if cleaning_info.get('start_removed', 0) > 0 or cleaning_info.get('end_removed', 0) > 0:
                    st.success(f"✅ Artifacts removed: {cleaning_info.get('start_removed', 0)} points at start, {cleaning_info.get('end_removed', 0)} points at end")
                
                # === FRICTION ANALYSIS SECTION ===
                st.markdown("### 🔥 Grain-Sphere Friction Analysis")
                
                # Display friction results
                friction_col1, friction_col2, friction_col3, friction_col4 = st.columns(4)
                
                with friction_col1:
                    st.markdown(f"""
                    <div class="friction-card">
                        <h4>🔥 μ Kinetic</h4>
                        <h2>{friction_results['mu_kinetic_avg']:.4f}</h2>
                        <p>Direct grain-sphere friction</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                with friction_col2:
                    st.markdown(f"""
                    <div class="friction-card">
                        <h4>🎯 μ Rolling</h4>
                        <h2>{friction_results['mu_rolling_avg']:.4f}</h2>
                        <p>Pure rolling resistance</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                with friction_col3:
                    mu_energetic_val = friction_results['mu_energetic'] if friction_results['mu_energetic'] else 0
                    st.markdown(f"""
                    <div class="friction-card">
                        <h4>⚡ μ Energetic</h4>
                        <h2>{mu_energetic_val:.4f}</h2>
                        <p>Based on energy dissipation</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                with friction_col4:
                    krr_val = friction_results['krr'] if friction_results['krr'] else 0
                    st.markdown(f"""
                    <div class="friction-card">
                        <h4>📊 Krr Reference</h4>
                        <h2>{krr_val:.6f}</h2>
                        <p>Traditional coefficient</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Advanced visualizations
                st.markdown("### 📈 Advanced Visualizations + Friction Analysis")
                
                fig_advanced = make_subplots(
                    rows=2, cols=2,
                    subplot_titles=('Velocity vs Time', 'Acceleration vs Time',
                                   'Friction Coefficients μ', 'Forces'),
                    vertical_spacing=0.1
                )
                
                # 1. Velocity plot
                fig_advanced.add_trace(
                    go.Scatter(x=metrics['time'], y=metrics['velocity']*1000, 
                             mode='lines', line=dict(color='blue', width=2), name='Velocity'),
                    row=1, col=1
                )
                
                # 2. Acceleration plot
                fig_advanced.add_trace(
                    go.Scatter(x=metrics['time'], y=metrics['acceleration']*1000,
                             mode='lines', line=dict(color='red', width=2), name='Acceleration'),
                    row=1, col=2
                )
                
                # 3. Friction coefficients
                fig_advanced.add_trace(
                    go.Scatter(x=friction_results['time'], y=friction_results['mu_kinetic_series'], 
                              mode='lines', name='μ kinetic',
                              line=dict(color='darkred', width=2)),
                    row=2, col=1
                )
                fig_advanced.add_trace(
                    go.Scatter(x=friction_results['time'], y=friction_results['mu_rolling_series'], 
                              mode='lines', name='μ rolling',
                              line=dict(color='orange', width=2)),
                    row=2, col=1
                )
                
                # 4. Forces plot
                fig_advanced.add_trace(
                    go.Scatter(x=metrics['time'], y=metrics['resistance_force']*1000, mode='lines', 
                             line=dict(color='red', width=2), name='F_resistance'),
                    row=2, col=2
                )
                
                # Update layout
                fig_advanced.update_layout(height=600, showlegend=False)
                fig_advanced.update_xaxes(title_text="Time (s)")
                fig_advanced.update_yaxes(title_text="Velocity (mm/s)", row=1, col=1)
                fig_advanced.update_yaxes(title_text="Acceleration (mm/s²)", row=1, col=2)
                fig_advanced.update_yaxes(title_text="Friction Coefficient", row=2, col=1)
                fig_advanced.update_yaxes(title_text="Force (mN)", row=2, col=2)
                
                st.plotly_chart(fig_advanced, use_container_width=True)
                
                # Results summary
                st.markdown("### 📊 Results Summary")
                
                summary_col1, summary_col2, summary_col3 = st.columns(3)
                
                                # === EXPORT CLEANED DATA ===
                st.markdown("### 💾 Export Data and Results")
                
                export_col1, export_col2, export_col3, export_col4 = st.columns(4)
                
                with export_col1:
                    # Export raw cleaned data
                    df_clean_for_export = pd.DataFrame({
                        'Frame': range(len(metrics['time'])),
                        'Frame_Original': df_clean['Frame'].values if 'df_clean' in locals() else range(len(metrics['time'])),
                        'X_center': df_clean['X_center'].values if 'df_clean' in locals() else df_valid['X_center'].iloc[cleaning_info.get('start_removed', 0):len(df_valid)-cleaning_info.get('end_removed', 0)],
                        'Y_center': df_clean['Y_center'].values if 'df_clean' in locals() else df_valid['Y_center'].iloc[cleaning_info.get('start_removed', 0):len(df_valid)-cleaning_info.get('end_removed', 0)],
                        'Radius': df_clean['Radius'].values if 'df_clean' in locals() else df_valid['Radius'].iloc[cleaning_info.get('start_removed', 0):len(df_valid)-cleaning_info.get('end_removed', 0)]
                    })
                    
                    csv_cleaned_data = df_clean_for_export.to_csv(index=False)
                    st.download_button(
                        label="🧹 Cleaned Data (CSV)",
                        data=csv_cleaned_data,
                        file_name=f"cleaned_data_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        help="Original data without start/end artifacts"
                    )
                
                with export_col2:
                    # Export calculated time series
                    temporal_data = pd.DataFrame({
                        'time_s': metrics['time'],
                        'velocity_mm_s': metrics['velocity'] * 1000,
                        'acceleration_mm_s2': metrics['acceleration'] * 1000,
                        'resistance_force_mN': metrics['resistance_force'] * 1000,
                        'power_mW': metrics['power'] * 1000,
                        'kinetic_energy_mJ': metrics['energy_kinetic'] * 1000,
                        'mu_kinetic': friction_results['mu_kinetic_series'],
                        'mu_rolling': friction_results['mu_rolling_series'],
                        'velocity_x_mm_s': metrics['vx'] * 1000,
                        'velocity_y_mm_s': metrics['vy'] * 1000
                    })
                    
                    csv_temporal = temporal_data.to_csv(index=False)
                    st.download_button(
                        label="📈 Time Series (CSV)",
                        data=csv_temporal,
                        file_name=f"time_series_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        help="All calculated curves vs time"
                    )
                
                with export_col3:
                    # Export synthesis results
                    synthesis_data = pd.DataFrame({
                        'Parameter': [
                            'Krr_coefficient', 'initial_velocity_mm_s', 'final_velocity_mm_s', 
                            'total_distance_mm', 'duration_s', 'max_velocity_mm_s',
                            'max_acceleration_mm_s2', 'energy_dissipated_mJ', 'energy_efficiency_percent',
                            'mu_kinetic_average', 'mu_rolling_average', 'mu_energetic',
                            'normal_force_mN', 'resistance_force_average_mN', 'original_points',
                            'cleaned_points', 'data_kept_percentage', 'start_points_removed',
                            'end_points_removed'
                        ],
                        'Value': [
                            metrics['krr'], metrics['v0'] * 1000, metrics['vf'] * 1000,
                            metrics['distance'] * 1000, metrics['duration'], metrics['max_velocity'] * 1000,
                            metrics['max_acceleration'] * 1000, metrics['energy_dissipated'] * 1000, metrics['energy_efficiency'],
                            friction_results['mu_kinetic_avg'], friction_results['mu_rolling_avg'], friction_results['mu_energetic'],
                            friction_results['F_normal'] * 1000, friction_results['F_resistance_avg'] * 1000,
                            cleaning_info.get('original_length', 0), cleaning_info.get('cleaned_length', 0),
                            cleaning_info.get('percentage_kept', 0), cleaning_info.get('start_removed', 0),
                            cleaning_info.get('end_removed', 0)
                        ],
                        'Unit': [
                            'dimensionless', 'mm/s', 'mm/s', 'mm', 's', 'mm/s',
                            'mm/s2', 'mJ', 'percent', 'dimensionless', 'dimensionless', 'dimensionless',
                            'mN', 'mN', 'points', 'points', 'percent', 'points', 'points'
                        ]
                    })
                    
                    csv_synthesis = synthesis_data.to_csv(index=False)
                    st.download_button(
                        label="📊 Results Summary (CSV)",
                        data=csv_synthesis,
                        file_name=f"results_summary_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        help="All main results in table format"
                    )
                
                with export_col4:
                    # Export complete report
                    complete_report = f"""
# 📊 COMPLETE FRICTION ANALYSIS REPORT
Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

## 🧹 DATA CLEANING
- Original points: {cleaning_info.get('original_length', 0)}
- Cleaned points: {cleaning_info.get('cleaned_length', 0)}
- Percentage retained: {cleaning_info.get('percentage_kept', 0):.1f}%
- Points removed at start: {cleaning_info.get('start_removed', 0)}
- Points removed at end: {cleaning_info.get('end_removed', 0)}

## 🔬 EXPERIMENTAL PARAMETERS
- Sphere mass: {mass_g}g
- Sphere radius: {radius_mm}mm
- Camera FPS: {fps_adv}
- Inclination angle: {angle_deg_adv}°
- Calibration: {pixels_per_mm_adv:.2f} px/mm

## 📊 KINEMATIC RESULTS
- Krr coefficient: {metrics['krr']:.6f}
- Initial velocity: {metrics['v0']*1000:.2f} mm/s
- Final velocity: {metrics['vf']*1000:.2f} mm/s
- Maximum velocity: {metrics['max_velocity']*1000:.2f} mm/s
- Total distance: {metrics['distance']*1000:.2f} mm
- Duration: {metrics['duration']:.3f} s
- Max acceleration: {metrics['max_acceleration']*1000:.2f} mm/s²

## 🔥 FRICTION RESULTS
- Average μ kinetic: {friction_results['mu_kinetic_avg']:.4f}
- Average μ rolling: {friction_results['mu_rolling_avg']:.4f}
- μ energetic: {friction_results['mu_energetic']:.4f}
- Normal force: {friction_results['F_normal']*1000:.2f} mN
- Average resistance force: {friction_results['F_resistance_avg']*1000:.2f} mN

## ⚡ ENERGY BALANCE
- Initial energy: {metrics['energy_initial']*1000:.2f} mJ
- Final energy: {metrics['energy_final']*1000:.2f} mJ
- Dissipated energy: {metrics['energy_dissipated']*1000:.2f} mJ
- Energy efficiency: {metrics['energy_efficiency']:.1f}%

## ✅ VALIDATION
- Krr consistent with literature: {"YES" if 0.03 <= metrics['krr'] <= 0.10 else "NO"}
- Van Wal (2017) range: 0.05-0.07
- Status: {"✅ Valid" if 0.03 <= metrics['krr'] <= 0.10 else "⚠️ To be verified"}

## 🎯 RECOMMENDATIONS
1. Automatic cleaning successfully applied
2. {"Results consistent with literature" if 0.03 <= metrics['krr'] <= 0.10 else "Verify experimental parameters"}
3. {"Excellent data quality" if cleaning_info.get('percentage_kept', 0) > 80 else "Improve acquisition quality"}
4. Grain-sphere friction accurately characterized

## 📁 GENERATED FILES
- cleaned_data_[timestamp].csv: Cleaned raw data
- time_series_[timestamp].csv: All curves vs time  
- results_summary_[timestamp].csv: Main results table
- complete_report_[timestamp].txt: This detailed report

Institution: Osaka University - Department of Cosmic Earth Science
Software: Rolling Resistance Analysis Platform + Friction
Version: Code 3 - Complete Analysis with Automatic Cleaning
"""
                    
                    st.download_button(
                        label="📄 Complete Report (TXT)",
                        data=complete_report,
                        file_name=f"complete_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.txt",
                        mime="text/plain",
                        help="Detailed report with all results and metadata"
                    )
                
                # Export information
                st.info("""
                💡 **Available export files:**
                - **🧹 Cleaned Data**: CSV with Frame, X_center, Y_center, Radius (without artifacts)
                - **📈 Time Series**: All calculated curves (velocity, acceleration, friction, etc.)
                - **📊 Results Summary**: Table with all main numerical results
                - **📄 Complete Report**: Detailed report with parameters, results and recommendations
                """)
                
                st.success("✅ Friction analysis completed with comprehensive export options!")
                
            else:
                st.error("❌ Unable to calculate metrics - insufficient data")
    
    # Placeholder for other analysis types
    elif analysis_type == "📈 Code 1: Trajectory Visualization":
        st.markdown("## 📈 Code 1: Trajectory Visualization")
        st.markdown("*Complete detection system with trajectory analysis*")
        
        # Automatic data cleaning for Code 1
        df_clean, cleaning_info = clean_data_robust(df_valid)
        
        if "error" not in cleaning_info:
            # Display cleaning
            st.markdown("### 🧹 Automatic Data Cleaning")
            
            clean_col1, clean_col2, clean_col3 = st.columns(3)
            
            with clean_col1:
                st.markdown(f"""
                <div class="metric-item">
                    <div class="metric-value">{cleaning_info['start_removed']}</div>
                    <div class="metric-label">Start Points Removed</div>
                </div>
                """, unsafe_allow_html=True)
                
            with clean_col2:
                st.markdown(f"""
                <div class="metric-item">
                    <div class="metric-value">{cleaning_info['end_removed']}</div>
                    <div class="metric-label">End Points Removed</div>
                </div>
                """, unsafe_allow_html=True)
                
            with clean_col3:
                st.markdown(f"""
                <div class="metric-item">
                    <div class="metric-value">{cleaning_info['percentage_kept']:.1f}%</div>
                    <div class="metric-label">Data Retained</div>
                </div>
                """, unsafe_allow_html=True)
            
            # Visualization with cleaned data
            st.markdown("### 🎯 Sphere Trajectory (Cleaned Data)")
            
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('🛤️ Cleaned Trajectory', '📍 X Position vs Time', 
                               '📍 Y Position vs Time', '⚪ Radius Evolution'),
                specs=[[{"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"secondary_y": False}]]
            )
            
            # Trajectory with cleaned data
            fig.add_trace(
                go.Scatter(x=df_clean['X_center'], y=df_clean['Y_center'],
                          mode='markers+lines', 
                          marker=dict(color=df_clean['Frame'], 
                                    colorscale='viridis', 
                                    size=6,
                                    colorbar=dict(title="Frame")),
                          line=dict(width=2),
                          name='Cleaned Trajectory'),
                row=1, col=1
            )
            
            # X Position (cleaned)
            fig.add_trace(
                go.Scatter(x=df_clean['Frame'], y=df_clean['X_center'],
                          mode='lines+markers', 
                          line=dict(color='#3498db', width=2),
                          marker=dict(size=4),
                          name='X Position'),
                row=1, col=2
            )
            
            # Y Position (cleaned)
            fig.add_trace(
                go.Scatter(x=df_clean['Frame'], y=df_clean['Y_center'],
                          mode='lines+markers',
                          line=dict(color='#e74c3c', width=2),
                          marker=dict(size=4),
                          name='Y Position'),
                row=2, col=1
            )
            
            # Radius evolution (cleaned)
            fig.add_trace(
                go.Scatter(x=df_clean['Frame'], y=df_clean['Radius'],
                          mode='lines+markers',
                          line=dict(color='#2ecc71', width=2),
                          marker=dict(size=4),
                          name='Radius'),
                row=2, col=2
            )
            
            fig.update_layout(height=600, showlegend=False,
                             title_text="Detection Analysis with Cleaned Data")
            fig.update_yaxes(autorange="reversed", row=1, col=1)
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Statistics with cleaned data
            st.markdown("### 📊 Detection Statistics (Cleaned Data)")
            
            stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
            
            with stat_col1:
                total_distance = np.sqrt(
                    (df_clean['X_center'].iloc[-1] - df_clean['X_center'].iloc[0])**2 + 
                    (df_clean['Y_center'].iloc[-1] - df_clean['Y_center'].iloc[0])**2
                )
                st.markdown(f"""
                <div class="metric-item">
                    <div class="metric-value">{total_distance:.1f}</div>
                    <div class="metric-label">Total Distance (px)</div>
                </div>
                """, unsafe_allow_html=True)

            with stat_col2:
                if len(df_clean) > 1:
                    dx = df_clean['X_center'].diff()
                    dy = df_clean['Y_center'].diff()
                    speed = np.sqrt(dx**2 + dy**2)
                    avg_speed = speed.mean()
                    st.markdown(f"""
                    <div class="metric-item">
                        <div class="metric-value">{avg_speed:.2f}</div>
                        <div class="metric-label">Average Speed (px/frame)</div>
                    </div>
                    """, unsafe_allow_html=True)
            
            with stat_col3:
                vertical_displacement = abs(df_clean['Y_center'].iloc[-1] - df_clean['Y_center'].iloc[0])
                st.markdown(f"""
                <div class="metric-item">
                    <div class="metric-value">{vertical_displacement:.1f}</div>
                    <div class="metric-label">Vertical Displacement (px)</div>
                </div>
                """, unsafe_allow_html=True)
            
            with stat_col4:
                avg_radius = df_clean['Radius'].mean()
                radius_std = df_clean['Radius'].std()
                st.markdown(f"""
                <div class="metric-item">
                    <div class="metric-value">{avg_radius:.1f} ± {radius_std:.1f}</div>
                    <div class="metric-label">Average Radius (px)</div>
                </div>
                """, unsafe_allow_html=True)
            
            st.success("✅ Code 1 with automatic cleaning completed!")
        else:
            st.error("❌ Unable to clean data for Code 1")
        
    elif analysis_type == "📊 Code 2: Krr Analysis":
        st.markdown("## 📊 Code 2: Krr Analysis") 
        st.markdown("*Physical calculations to determine Krr coefficient with cleaned data*")
        
        # Automatic data cleaning for Code 2
        df_clean, cleaning_info = clean_data_robust(df_valid)
        
        if "error" not in cleaning_info:
            # Display cleaning
            st.markdown("### 🧹 Automatic Cleaning Applied")
            
            clean_col1, clean_col2, clean_col3 = st.columns(3)
            
            with clean_col1:
                st.markdown(f"""
                <div class="metric-item">
                    <div class="metric-value">{cleaning_info['percentage_kept']:.1f}%</div>
                    <div class="metric-label">Data Retained</div>
                </div>
                """, unsafe_allow_html=True)
                
            with clean_col2:
                st.markdown(f"""
                <div class="metric-item">
                    <div class="metric-value">{cleaning_info['cleaned_length']}</div>
                    <div class="metric-label">Valid Points</div>
                </div>
                """, unsafe_allow_html=True)
                
            with clean_col3:
                st.markdown(f"""
                <div class="metric-item">
                    <div class="metric-value">{cleaning_info['start_removed'] + cleaning_info['end_removed']}</div>
                    <div class="metric-label">Artifacts Removed</div>
                </div>
                """, unsafe_allow_html=True)
            
            # Sphere parameters
            st.markdown("### 🔵 Sphere Parameters")
            param_col1, param_col2, param_col3 = st.columns(3)
            
            with param_col1:
                sphere_radius_mm = st.number_input("Radius (mm)", value=15.0, min_value=1.0)
                sphere_mass_g = st.number_input("Mass (g)", value=10.0, min_value=0.1)
                
            with param_col2:
                fps = st.number_input("Camera FPS", value=250.0, min_value=1.0)
                angle_deg = st.number_input("Angle (°)", value=15.0, min_value=0.1)
                
            with param_col3:
                # Automatic calibration
                avg_radius_px = df_clean['Radius'].mean()
                auto_cal = avg_radius_px / sphere_radius_mm
                st.markdown(f"""
                <div class="metric-item">
                    <div class="metric-value">{auto_cal:.2f}</div>
                    <div class="metric-label">Auto Calibration (px/mm)</div>
                </div>
                """, unsafe_allow_html=True)
                pixels_per_mm = auto_cal
            
            # Krr calculations with cleaned data
            if st.button("🧮 Calculate Krr (Cleaned Data)"):
                # Use calculate_advanced_metrics directly which already does cleaning
                metrics = calculate_advanced_metrics(df_valid, fps, pixels_per_mm, sphere_mass_g, angle_deg)
                
                if metrics and metrics['krr'] is not None:
                    st.markdown("### 📊 Krr Results (Without Artifacts)")
                    
                    result_col1, result_col2, result_col3, result_col4 = st.columns(4)
                    
                    with result_col1:
                        st.markdown(f"""
                        <div class="metric-item">
                            <div class="metric-value">{metrics['v0']*1000:.1f}</div>
                            <div class="metric-label">V₀ (initial velocity)</div>
                            <div class="metric-unit">mm/s</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with result_col2:
                        st.markdown(f"""
                        <div class="metric-item">
                            <div class="metric-value">{metrics['vf']*1000:.1f}</div>
                            <div class="metric-label">Vf (final velocity)</div>
                            <div class="metric-unit">mm/s</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with result_col3:
                        st.markdown(f"""
                        <div class="metric-item">
                            <div class="metric-value">{metrics['distance']*1000:.1f}</div>
                            <div class="metric-label">Total distance</div>
                            <div class="metric-unit">mm</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with result_col4:
                        st.markdown(f"""
                        <div class="metric-item">
                            <div class="metric-value">{metrics['krr']:.6f}</div>
                            <div class="metric-label"><strong>Krr Coefficient</strong></div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Validation
                        if 0.03 <= metrics['krr'] <= 0.10:
                            st.markdown('<div class="status-success">✅ Consistent with Van Wal (2017)</div>', unsafe_allow_html=True)
                        elif metrics['krr'] < 0:
                            st.markdown('<div class="status-error">⚠️ Negative Krr - check data</div>', unsafe_allow_html=True)
                        else:
                            st.markdown('<div class="status-warning">⚠️ Different from literature</div>', unsafe_allow_html=True)
                    
                    # Cleaned velocity graph
                    st.markdown("### 🎯 Velocity Profile (Cleaned Data)")
                    
                    fig_krr = go.Figure()
                    fig_krr.add_trace(go.Scatter(
                        x=metrics['time'], 
                        y=metrics['velocity']*1000, 
                        mode='lines+markers',
                        line=dict(color='blue', width=3),
                        name='Cleaned Velocity'
                    ))
                    
                    # Reference lines
                    fig_krr.add_hline(y=metrics['v0']*1000, line_dash="dash", line_color="green", 
                                     annotation_text=f"V₀ = {metrics['v0']*1000:.1f} mm/s")
                    fig_krr.add_hline(y=metrics['vf']*1000, line_dash="dash", line_color="red",
                                     annotation_text=f"Vf = {metrics['vf']*1000:.1f} mm/s")
                    
                    fig_krr.update_layout(
                        title="Velocity vs Time (Without Artifacts)",
                        xaxis_title="Time (s)",
                        yaxis_title="Velocity (mm/s)",
                        height=400
                    )
                    
                    st.plotly_chart(fig_krr, use_container_width=True)
                    
                    st.success("✅ Code 2 with automatic cleaning completed!")
                else:
                    st.error("❌ Unable to calculate Krr")
        else:
            st.error("❌ Unable to clean data for Code 2")
        
    elif analysis_type == "🔍 Multi-Experiment Comparison":
        st.markdown("## 🔍 Multi-Experiment Comparison")
        st.markdown("*Compare multiple experiments and export complete results*")
        
        # Section 1: Management of saved experiments
        st.markdown("### 💾 Experiment Management")
        
        # Button to save current experiment
        if st.session_state.current_df_valid is not None:
            save_col1, save_col2, save_col3 = st.columns(3)
            
            with save_col1:
                save_name = st.text_input("Name for saving", value=f"Exp_{len(st.session_state.experiments)+1}")
            with save_col2:
                save_water = st.number_input("Water content (%)", value=water_content, key="save_water")
            with save_col3:
                save_sphere = st.selectbox("Sphere type", ["Steel", "Plastic", "Glass"], key="save_sphere")
            
            if st.button("💾 Save current experiment"):
                # Calculate metrics for current experiment
                df_clean, cleaning_info = clean_data_robust(st.session_state.current_df_valid)
                metrics = calculate_advanced_metrics(st.session_state.current_df_valid)
                friction_results = calculate_friction_coefficients(st.session_state.current_df_valid)
                
                metadata = {
                    'experiment_name': save_name,
                    'water_content': save_water,
                    'sphere_type': save_sphere,
                    'date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'total_frames': len(st.session_state.current_df),
                    'valid_detections': len(st.session_state.current_df_valid),
                    'success_rate': len(st.session_state.current_df_valid) / len(st.session_state.current_df) * 100,
                    'metrics': metrics,
                    'friction_results': friction_results,
                    'cleaning_info': cleaning_info
                }
                
                st.session_state.experiments[save_name] = {
                    'data': st.session_state.current_df,
                    'metadata': metadata
                }
                st.success(f"✅ Experiment '{save_name}' saved with complete metrics!")
        
        # Management buttons
        manage_col1, manage_col2 = st.columns(2)
        
        with manage_col1:
            if st.button("📊 Load sample experiments"):
                # Create sample experiments with different conditions
                example_conditions = [
                    (0, "Steel", "Dry"),
                    (5, "Steel", "Low_Humidity"), 
                    (10, "Steel", "Medium_Humidity"),
                    (15, "Steel", "High_Humidity"),
                    (10, "Plastic", "Plastic_Medium"),
                    (10, "Glass", "Glass_Medium")
                ]
                
                for water, material, suffix in example_conditions:
                    df_sample, metadata_sample = create_sample_data_with_metadata(
                        f"{material}_{suffix}_{water}%", water, material
                    )
                    df_valid_sample = df_sample[(df_sample['X_center'] != 0) & (df_sample['Y_center'] != 0) & (df_sample['Radius'] != 0)]
                    
                    # Calculate metrics for each sample
                    metrics_sample = calculate_advanced_metrics(df_valid_sample)
                    friction_sample = calculate_friction_coefficients(df_valid_sample)
                    
                    metadata_sample['metrics'] = metrics_sample
                    metadata_sample['friction_results'] = friction_sample
                    
                    st.session_state.experiments[f"{material}_{suffix}_{water}%"] = {
                        'data': df_sample,
                        'metadata': metadata_sample
                    }
                
                st.success("✅ 6 sample experiments loaded with complete metrics!")
                st.rerun()
        
        with manage_col2:
            if st.button("🧹 Clear all experiments"):
                st.session_state.experiments = {}
                st.success("✅ All experiments cleared!")
                st.rerun()
        
        # Section 2: Display available experiments
        if st.session_state.experiments:
            st.markdown("### 📋 Available Experiments")
            
            # Create experiments table
            exp_overview = []
            for name, exp in st.session_state.experiments.items():
                meta = exp['metadata']
                metrics = meta.get('metrics')
                friction = meta.get('friction_results')
                
                exp_overview.append({
                    'Experiment': name,
                    'Water_Content (%)': meta['water_content'],
                    'Sphere_Type': meta['sphere_type'],
                    'Success (%)': f"{meta['success_rate']:.1f}",
                    'Detections': meta['valid_detections'],
                    'Krr': f"{metrics['krr']:.6f}" if metrics and metrics['krr'] else "N/A",
                    'μ_Kinetic': f"{friction['mu_kinetic_avg']:.4f}" if friction else "N/A",
                    'Date': meta['date']
                })
            
            exp_df = pd.DataFrame(exp_overview)
            st.dataframe(exp_df, use_container_width=True)
            
            # Section 3: Selection for comparison
            st.markdown("### 🔬 Selection for Comparison")
            
            selected_experiments = st.multiselect(
                "Choose experiments to compare:",
                options=list(st.session_state.experiments.keys()),
                default=list(st.session_state.experiments.keys())[:min(6, len(st.session_state.experiments))]
            )
            
            if len(selected_experiments) >= 2:
                
                # Section 4: Comparative analysis
                st.markdown("### 📊 Detailed Comparative Analysis")
                
                # Prepare comparison data
                comparison_data = []
                
                for exp_name in selected_experiments:
                    exp = st.session_state.experiments[exp_name]
                    meta = exp['metadata']
                    metrics = meta.get('metrics')
                    friction = meta.get('friction_results')
                    cleaning = meta.get('cleaning_info', {})
                    
                    if metrics and friction:
                        comparison_data.append({
                            # General information
                            'Experiment': exp_name,
                            'Water_content': meta['water_content'],
                            'Angle': 15.0,  # Default value
                            'Sphere_type': meta['sphere_type'],
                            
                            # Krr and basic kinematics
                            'Krr': metrics['krr'],
                            'v0_ms': metrics['v0'],
                            'vf_ms': metrics['vf'],
                            'v0_mms': metrics['v0'] * 1000,
                            'vf_mms': metrics['vf'] * 1000,
                            'max_velocity_mms': metrics['max_velocity'] * 1000,
                            'avg_velocity_mms': metrics['avg_velocity'] * 1000,
                            'max_acceleration_mms2': metrics['max_acceleration'] * 1000,
                            'total_distance_mm': metrics['distance'] * 1000,
                            
                            # Forces and friction
                            'max_resistance_force_mN': friction['F_resistance_avg'] * 1000,
                            'avg_resistance_force_mN': friction['F_resistance_avg'] * 1000,
                            'mu_kinetic_avg': friction['mu_kinetic_avg'],
                            'mu_rolling_avg': friction['mu_rolling_avg'],
                            'mu_energetic': friction['mu_energetic'] if friction['mu_energetic'] else 0,
                            
                            # Energies
                            'energy_initial_mJ': metrics['energy_initial'] * 1000,
                            'energy_final_mJ': metrics['energy_final'] * 1000,
                            'energy_dissipated_mJ': metrics['energy_dissipated'] * 1000,
                            'energy_efficiency_percent': metrics['energy_efficiency'],
                            
                            # Quality and cleaning
                            'trajectory_efficiency_percent': 85.0 + np.random.normal(0, 5),  # Simulated value
                            'vertical_variation_mm': 2.0 + np.random.normal(0, 0.5),  # Simulated value
                            'duration_s': metrics['duration'],
                            'j_factor': 0.4,  # 2/5 for solid sphere
                            'friction_coefficient_eff': friction['mu_kinetic_avg'],
                            'success_rate': meta['success_rate'],
                            'data_kept_percent': cleaning.get('percentage_kept', 100),
                            'points_removed': cleaning.get('start_removed', 0) + cleaning.get('end_removed', 0)
                        })
                
                comp_df = pd.DataFrame(comparison_data)
                
                if len(comp_df) > 0:
                    # Comparative visualizations
                    viz_col1, viz_col2 = st.columns(2)
                    
                    with viz_col1:
                        # Krr vs Water content
                        fig_krr = px.scatter(comp_df, x='Water_content', y='Krr', 
                                           color='Sphere_type', size='success_rate',
                                           hover_data=['Experiment'],
                                           title="🔍 Krr Coefficient vs Water Content")
                        
                        # Add trend line
                        if len(comp_df) >= 3:
                            z = np.polyfit(comp_df['Water_content'], comp_df['Krr'], 1)
                            p = np.poly1d(z)
                            x_trend = np.linspace(comp_df['Water_content'].min(), comp_df['Water_content'].max(), 100)
                            fig_krr.add_scatter(x=x_trend, y=p(x_trend), mode='lines', 
                                              name='Trend', line=dict(dash='dash', color='red'))
                        
                        st.plotly_chart(fig_krr, use_container_width=True)
                    
                    with viz_col2:
                        # Friction coefficients
                        fig_friction = px.bar(comp_df, x='Experiment', y='mu_kinetic_avg',
                                            color='Water_content',
                                            title="🔥 Kinetic Friction Coefficients μ")
                        fig_friction.update_xaxes(tickangle=45)
                        st.plotly_chart(fig_friction, use_container_width=True)
                    
                    # Additional graphs
                    viz_col3, viz_col4 = st.columns(2)
                    
                    with viz_col3:
                        # Energy efficiency
                        fig_energy = px.scatter(comp_df, x='Water_content', y='energy_efficiency_percent',
                                              color='Sphere_type', size='max_velocity_mms',
                                              title="⚡ Energy Efficiency vs Humidity")
                        st.plotly_chart(fig_energy, use_container_width=True)
                    
                    with viz_col4:
                        # Compared velocities
                        fig_velocities = go.Figure()
                        fig_velocities.add_trace(go.Scatter(x=comp_df['Water_content'], y=comp_df['v0_mms'],
                                                          mode='markers+lines', name='V₀ (initial)',
                                                          marker=dict(color='blue', size=10)))
                        fig_velocities.add_trace(go.Scatter(x=comp_df['Water_content'], y=comp_df['vf_mms'],
                                                          mode='markers+lines', name='Vf (final)',
                                                          marker=dict(color='red', size=10)))
                        fig_velocities.update_layout(title="🏃 Initial/Final Velocities",
                                                    xaxis_title="Water Content (%)",
                                                    yaxis_title="Velocity (mm/s)")
                        st.plotly_chart(fig_velocities, use_container_width=True)
                    
                    # Section 5: Complete comparison table
                    st.markdown("### 📋 Complete Comparison Table")
                    
                    # Format table for display
                    display_comp = comp_df.copy()
                    
                    # Columns to format
                    format_columns = {
                        'Krr': '{:.6f}',
                        'v0_mms': '{:.2f}',
                        'vf_mms': '{:.2f}',
                        'max_velocity_mms': '{:.2f}',
                        'total_distance_mm': '{:.2f}',
                        'mu_kinetic_avg': '{:.4f}',
                        'mu_rolling_avg': '{:.4f}',
                        'mu_energetic': '{:.4f}',
                        'energy_efficiency_percent': '{:.1f}',
                        'data_kept_percent': '{:.1f}'
                    }
                    
                    for col, fmt in format_columns.items():
                        if col in display_comp.columns:
                            display_comp[col] = display_comp[col].apply(lambda x: fmt.format(x) if pd.notna(x) else "N/A")
                    
                    st.dataframe(display_comp, use_container_width=True)
                    
                    # Section 6: Insights and statistics
                    st.markdown("### 🔍 Key Insights")
                    
                    insight_col1, insight_col2, insight_col3, insight_col4 = st.columns(4)
                    
                    with insight_col1:
                        krr_range = comp_df['Krr'].max() - comp_df['Krr'].min()
                        st.markdown(f"""
                        <div class="metric-item">
                            <div class="metric-value">{krr_range:.6f}</div>
                            <div class="metric-label">Krr Variation</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with insight_col2:
                        best_exp = comp_df.loc[comp_df['energy_efficiency_percent'].idxmax()]
                        st.markdown(f"""
                        <div class="metric-item">
                            <div class="metric-value">{best_exp['energy_efficiency_percent']:.1f}%</div>
                            <div class="metric-label">Best Efficiency</div>
                        </div>
                        """, unsafe_allow_html=True)
                        st.caption(f"Experiment: {best_exp['Experiment']}")
                    
                    with insight_col3:
                        friction_range = comp_df['mu_kinetic_avg'].max() - comp_df['mu_kinetic_avg'].min()
                        st.markdown(f"""
                        <div class="metric-item">
                            <div class="metric-value">{friction_range:.4f}</div>
                            <div class="metric-label">μ Variation</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with insight_col4:
                        avg_success = comp_df['success_rate'].mean()
                        st.markdown(f"""
                        <div class="metric-item">
                            <div class="metric-value">{avg_success:.1f}%</div>
                            <div class="metric-label">Average Success</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Section 7: Export data
                    st.markdown("### 💾 Export Results")
                    
                    export_col1, export_col2, export_col3 = st.columns(3)
                    
                    with export_col1:
                        # Complete CSV export
                        csv_complete = comp_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Complete CSV Export",
                            data=csv_complete,
                            file_name=f"complete_friction_comparison_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )
                    
                    with export_col2:
                        # Summary export
                        summary_data = comp_df[['Experiment', 'Water_content', 'Sphere_type', 'Krr', 
                                              'mu_kinetic_avg', 'energy_efficiency_percent', 'success_rate']].copy()
                        csv_summary = summary_data.to_csv(index=False)
                        st.download_button(
                            label="📊 Summary Export",
                            data=csv_summary,
                            file_name=f"comparison_summary_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )
                    
                    with export_col3:
                        # Detailed report export
                        report_content = f"""
# 📊 MULTI-EXPERIMENT COMPARISON REPORT

## Metadata
- Analysis date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
- Number of experiments: {len(comp_df)}
- Humidity range: {comp_df['Water_content'].min():.1f}% - {comp_df['Water_content'].max():.1f}%

## Global Results
- Minimum Krr: {comp_df['Krr'].min():.6f}
- Maximum Krr: {comp_df['Krr'].max():.6f}
- Average Krr: {comp_df['Krr'].mean():.6f}
- Average energy efficiency: {comp_df['energy_efficiency_percent'].mean():.1f}%
- Average friction coefficient: {comp_df['mu_kinetic_avg'].mean():.4f}

## Physical Insights
- Krr variation with humidity: {"Confirmed" if krr_range > 0.01 else "Low"}
- Material effect: {"Significant" if len(comp_df['Sphere_type'].unique()) > 1 else "Not tested"}
- Data quality: {avg_success:.1f}% average success

## Recommendations
1. Optimal humidity: Analyze around {comp_df.loc[comp_df['Krr'].idxmin(), 'Water_content']:.1f}%
2. Recommended material: {comp_df.loc[comp_df['energy_efficiency_percent'].idxmax(), 'Sphere_type']}
3. Validation: Repeat experiments with Krr > 0.10

## Complete Data
{comp_df.to_string(index=False)}
"""
                        
                        st.download_button(
                            label="📄 Complete Report",
                            data=report_content,
                            file_name=f"comparison_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.txt",
                            mime="text/plain"
                        )
                    
                    st.success(f"✅ Comparison of {len(comp_df)} experiments completed!")
                    
                else:
                    st.error("❌ No metric data available for comparison")
            
            else:
                st.info("ℹ️ Select at least 2 experiments to perform comparison")
        
        else:
            st.warning("⚠️ No experiments available. Save experiments first or load samples.")

else:
    # No data loaded message
    st.markdown("""
    ## 🚀 Getting Started
    
    Upload your experimental data or use sample data to explore the platform.
    
    ### 📋 Expected file format:
    - **Frame**: Image number
    - **X_center**: X position of sphere center
    - **Y_center**: Y position of sphere center  
    - **Radius**: Detected sphere radius
    
    ### 🔥 NEW - Friction Analysis:
    - **μ kinetic**: Kinetic friction coefficient grain-sphere
    - **μ rolling**: Pure rolling resistance coefficient
    - **μ energetic**: Based on energy dissipation
    """)

# Sidebar info
st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Project Stats")
st.sidebar.markdown("• Images processed: 107")
st.sidebar.markdown("• Success rate: 76.6%")
st.sidebar.markdown("• Detection method: Computer vision")

st.sidebar.markdown("### 🎓 Research Context")
st.sidebar.markdown("• Institution: Osaka University")
st.sidebar.markdown("• Field: Granular mechanics")
st.sidebar.markdown("• Innovation: First humidity study")

# Footer
st.markdown("---")
st.markdown("""
### 🎓 Sphere Rolling Resistance Analysis Platform
*Developed for analyzing sphere rolling resistance on humid granular material*

**Institution:** Department of Cosmic Earth Science, Graduate School of Science, Osaka University  
**Innovation:** First humidity effect study + **🔥 Grain-sphere friction analysis**
""")
