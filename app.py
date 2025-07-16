import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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

# Custom CSS following your vision
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem 2rem;
        border-radius: 10px;
        color: white;
        margin-bottom: 2rem;
    }
    .section-card {
        background: white;
        padding: 1.5rem;
        border-radius: 8px;
        border: 1px solid #e1e5e9;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .metric-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
        margin: 1rem 0;
    }
    .metric-item {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 6px;
        text-align: center;
        border: 1px solid #dee2e6;
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
    .metric-unit {
        font-size: 0.8rem;
        color: #6c757d;
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
    .project-info {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 6px;
        border: 1px solid #dee2e6;
        margin: 1rem 0;
    }
    .project-info h4 {
        margin-top: 0;
        color: #495057;
    }
    .analysis-results {
        background: white;
        border: 1px solid #dee2e6;
        border-radius: 8px;
        overflow: hidden;
    }
    .results-header {
        background: linear-gradient(90deg, #28a745, #20c997);
        color: white;
        padding: 1rem;
        margin: 0;
    }
    .results-content {
        padding: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)

# ==================== HELPER FUNCTIONS ====================

def create_sample_data_with_metadata(experiment_name="Sample", water_content=0.0, sphere_type="Steel"):
    """Creates sample data with experimental metadata"""
    frames = list(range(1, 108))
    data = []
    
    # Modify trajectory based on water content (simulation)
    water_effect = 1 + (water_content / 100) * 0.3  # Higher water content = more resistance
    
    for frame in frames:
        if frame < 9:
            data.append([frame, 0, 0, 0])
        elif frame in [30, 31]:
            data.append([frame, 0, 0, 0])
        else:
            # Adjust movement based on water content
            x = 1240 - (frame - 9) * 12 * water_effect + np.random.normal(0, 2)
            y = 680 + (frame - 9) * 0.5 + np.random.normal(0, 3)
            radius = 20 + np.random.normal(5, 3)
            radius = max(18, min(35, radius))
            data.append([frame, max(0, x), max(0, y), max(0, radius)])
    
    df = pd.DataFrame(data, columns=['Frame', 'X_center', 'Y_center', 'Radius'])
    
    # Add metadata
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

def calculate_advanced_metrics(df_valid, fps=250, pixels_per_mm=5.0, sphere_mass_g=10.0, angle_deg=15.0):
    """Calculate comprehensive kinematic and dynamic metrics"""
    if len(df_valid) < 10:
        return None
    
    # Convert to real units
    dt = 1 / fps
    mass_kg = sphere_mass_g / 1000
    angle_rad = np.radians(angle_deg)
    g = 9.81
    
    x_m = df_valid['X_center'].values / pixels_per_mm / 1000
    y_m = df_valid['Y_center'].values / pixels_per_mm / 1000
    
    # Time array
    t = np.arange(len(df_valid)) * dt
    
    # Calculate velocities and accelerations
    vx = np.gradient(x_m, dt)
    vy = np.gradient(y_m, dt)
    v_magnitude = np.sqrt(vx**2 + vy**2)
    
    # Accelerations
    ax = np.gradient(vx, dt)
    ay = np.gradient(vy, dt)
    acceleration = np.gradient(v_magnitude, dt)
    
    # Forces
    F_resistance = mass_kg * np.abs(acceleration)
    F_gravity = mass_kg * g * np.sin(angle_rad)
    
    # Energies
    E_kinetic = 0.5 * mass_kg * v_magnitude**2
    E_initial = E_kinetic[0] if len(E_kinetic) > 0 else 0
    E_final = E_kinetic[-1] if len(E_kinetic) > 0 else 0
    E_dissipated = E_initial - E_final
    
    # Power
    P_resistance = F_resistance * v_magnitude
    
    # Trajectory quality metrics
    y_variation = np.std(y_m) * 1000  # mm
    path_length = np.sum(np.sqrt(np.diff(x_m)**2 + np.diff(y_m)**2))
    straight_distance = np.sqrt((x_m[-1] - x_m[0])**2 + (y_m[-1] - y_m[0])**2)
    trajectory_efficiency = straight_distance / path_length if path_length > 0 else 0
    
    # Basic Krr calculation
    n_avg = min(3, len(v_magnitude)//4)
    v0 = np.mean(v_magnitude[:n_avg]) if len(v_magnitude) >= n_avg else v_magnitude[0]
    vf = np.mean(v_magnitude[-n_avg:]) if len(v_magnitude) >= n_avg else v_magnitude[-1]
    
    distances = np.sqrt(np.diff(x_m)**2 + np.diff(y_m)**2)
    total_distance = np.sum(distances)
    
    krr = (v0**2 - vf**2) / (2 * g * total_distance) if total_distance > 0 and v0 > vf else None
    
    return {
        # Basic metrics
        'krr': krr,
        'v0': v0,
        'vf': vf,
        'distance': total_distance,
        'duration': t[-1] - t[0],
        
        # Advanced kinematic metrics
        'max_velocity': np.max(v_magnitude),
        'avg_velocity': np.mean(v_magnitude),
        'max_acceleration': np.max(np.abs(acceleration)),
        'avg_acceleration': np.mean(np.abs(acceleration)),
        'initial_acceleration': np.abs(acceleration[0]) if len(acceleration) > 0 else 0,
        
        # Force and energy metrics
        'max_resistance_force': np.max(F_resistance),
        'avg_resistance_force': np.mean(F_resistance),
        'max_power': np.max(P_resistance),
        'avg_power': np.mean(P_resistance),
        'energy_initial': E_initial,
        'energy_final': E_final,
        'energy_dissipated': E_dissipated,
        'energy_efficiency': (E_final / E_initial * 100) if E_initial > 0 else 0,
        
        # Trajectory quality metrics
        'trajectory_efficiency': trajectory_efficiency * 100,
        'vertical_variation': y_variation,
        'path_length': path_length * 1000,  # mm
        
        # Time series for plotting
        'time': t,
        'velocity': v_magnitude,
        'acceleration': acceleration,
        'resistance_force': F_resistance,
        'power': P_resistance,
        'energy_kinetic': E_kinetic,
        'vx': vx,
        'vy': vy
    }

# ==================== MAIN APPLICATION ====================

# Header
st.markdown("""
<div class="main-header">
    <h1>📊 Plateforme d'Analyse de Résistance au Roulement des Sphères</h1>
    <p>Suite d'analyse complète pour la recherche en mécanique granulaire - Université d'Osaka</p>
</div>
""", unsafe_allow_html=True)

# Sidebar Configuration
st.sidebar.markdown("### 📋 Navigation")

# Project Stats (Fixed in sidebar)
st.sidebar.markdown("""
<div class="project-info">
    <h4>📊 Project Stats</h4>
    <ul style="margin: 0; padding-left: 1rem;">
        <li><strong>Images processed:</strong> 107</li>
        <li><strong>Success rate:</strong> 76.6%</li>
        <li><strong>Detection method:</strong> Computer vision</li>
        <li><strong>Research type:</strong> Experimental physics</li>
    </ul>
</div>
""", unsafe_allow_html=True)

# Research Context (Fixed in sidebar)
st.sidebar.markdown("""
<div class="project-info">
    <h4>🎓 Research Context</h4>
    <ul style="margin: 0; padding-left: 1rem;">
        <li><strong>Institution:</strong> Osaka University</li>
        <li><strong>Field:</strong> Granular mechanics</li>
        <li><strong>Innovation:</strong> First humidity study</li>
        <li><strong>Impact:</strong> Engineering applications</li>
    </ul>
</div>
""", unsafe_allow_html=True)

# Analysis Type Selection
analysis_type = st.sidebar.selectbox(
    "Sélectionnez le type d'analyse :",
    [
        "📈 Code 1 : Visualisation de Trajectoire",
        "📊 Code 2 : Analyse Krr",
        "🔬 Code 3 : Analyse Complète Avancée",
        "🔍 Comparaison Multi-Expériences",
        "🔄 Analyse de Reproductibilité",
        "🎯 Module de Prédiction",
        "📄 Rapport Auto-Généré"
    ]
)

# ==================== DATA LOADING SECTION ====================

# Data loading section (always visible)
st.markdown("## 📂 Chargement des Données")

# Create tabs for data input
tab1, tab2 = st.tabs(["📁 Upload Fichier", "🔬 Données d'Exemple"])

with tab1:
    col1, col2, col3 = st.columns(3)
    with col1:
        experiment_name = st.text_input("Nom de l'Expérience", value="Experiment_1")
    with col2:
        water_content = st.number_input("Teneur en Eau (%)", value=0.0, min_value=0.0, max_value=30.0)
    with col3:
        sphere_type = st.selectbox("Type de Sphère", ["Steel", "Plastic", "Glass", "Other"])
    
    uploaded_file = st.file_uploader(
        "Choisissez votre fichier CSV avec les données de détection", 
        type=['csv'],
        help="CSV avec colonnes: Frame, X_center, Y_center, Radius"
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
                st.success(f"✅ Fichier chargé: {len(df)} frames, {len(df_valid)} détections valides ({success_rate:.1f}%)")
                
                # Save experiment option
                if st.button("💾 Sauvegarder pour comparaison"):
                    metadata = {
                        'experiment_name': experiment_name,
                        'water_content': water_content,
                        'sphere_type': sphere_type,
                        'date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
                        'total_frames': len(df),
                        'valid_detections': len(df_valid),
                        'success_rate': success_rate
                    }
                    st.session_state.experiments[experiment_name] = {
                        'data': df,
                        'metadata': metadata
                    }
                    st.success(f"Expérience '{experiment_name}' sauvegardée!")
            else:
                st.error(f"❌ Colonnes manquantes: {[col for col in required_columns if col not in df.columns]}")
                
        except Exception as e:
            st.error(f"❌ Erreur: {str(e)}")

with tab2:
    col1, col2 = st.columns(2)
    with col1:
        sample_water = st.slider("Teneur en Eau d'Exemple (%)", 0.0, 25.0, 10.0, 0.5)
    with col2:
        sample_sphere = st.selectbox("Type de Sphère d'Exemple", ["Steel", "Plastic", "Glass"], key="sample_sphere")
    
    if st.button("🔬 Générer des données d'exemple"):
        df, metadata = create_sample_data_with_metadata(
            experiment_name=f"Sample_{sample_water}%", 
            water_content=sample_water, 
            sphere_type=sample_sphere
        )
        df_valid = df[(df['X_center'] != 0) & (df['Y_center'] != 0) & (df['Radius'] != 0)]
        
        st.session_state.current_df = df
        st.session_state.current_df_valid = df_valid
        st.success("📊 Données d'exemple générées!")

# Data overview (if data loaded)
if st.session_state.current_df is not None and st.session_state.current_df_valid is not None:
    df = st.session_state.current_df
    df_valid = st.session_state.current_df_valid
    
    # Quick overview
    st.markdown("""
    <div class="section-card">
        <h3>📊 Aperçu des Données Chargées</h3>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-item">
            <div class="metric-value">{len(df)}</div>
            <div class="metric-label">Frames Totales</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-item">
            <div class="metric-value">{len(df_valid)}</div>
            <div class="metric-label">Détections Valides</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        success_rate = len(df_valid) / len(df) * 100 if len(df) > 0 else 0
        st.markdown(f"""
        <div class="metric-item">
            <div class="metric-value">{success_rate:.1f}%</div>
            <div class="metric-label">Taux de Succès</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        avg_radius = df_valid['Radius'].mean() if len(df_valid) > 0 else 0
        st.markdown(f"""
        <div class="metric-item">
            <div class="metric-value">{avg_radius:.1f}</div>
            <div class="metric-label">Rayon Moyen (px)</div>
        </div>
        """, unsafe_allow_html=True)

# ==================== ANALYSIS SECTIONS ====================

if st.session_state.current_df_valid is not None and len(st.session_state.current_df_valid) > 0:
    df_valid = st.session_state.current_df_valid
    
    # ===== CODE 1: TRAJECTORY VISUALIZATION =====
    if analysis_type == "📈 Code 1 : Visualisation de Trajectoire":
        st.markdown("""
        <div class="analysis-results">
            <h2 class="results-header">📈 Code 1 : Détection et Visualisation de Trajectoire</h2>
            <div class="results-content">
        """, unsafe_allow_html=True)
        
        # Detection configuration
        st.markdown("### ⚙️ Configuration de Détection")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Paramètres de Taille**")
            minR = st.slider("Rayon minimum", 10, 30, 18)
            maxR = st.slider("Rayon maximum", 25, 50, 35)
            
        with col2:
            st.markdown("**Paramètres de Détection**")
            bw_threshold = st.slider("Seuil de détection", 1, 20, 8)
            min_score = st.slider("Score minimum", 20, 60, 40)
            
        with col3:
            st.markdown("**Paramètres de Forme**")
            circularity_min = st.slider("Circularité minimum", 0.1, 1.0, 0.5)
            max_movement = st.slider("Mouvement max", 50, 200, 120)
        
        # Main trajectory visualization
        st.markdown("### 🎯 Trajectoire de la Sphère Détectée")
        
        # Create 4-subplot visualization as shown in your image
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('🛤️ Trajectoire Complète', '📍 Position X vs Temps', 
                           '📍 Position Y vs Temps', '⚪ Évolution du Rayon'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Trajectory with color gradient
        fig.add_trace(
            go.Scatter(x=df_valid['X_center'], y=df_valid['Y_center'],
                      mode='markers+lines', 
                      marker=dict(color=df_valid['Frame'], 
                                colorscale='viridis', 
                                size=6,
                                colorbar=dict(title="Frame")),
                      line=dict(width=2),
                      name='Trajectoire'),
            row=1, col=1
        )
        
        # X Position
        fig.add_trace(
            go.Scatter(x=df_valid['Frame'], y=df_valid['X_center'],
                      mode='lines+markers', 
                      line=dict(color='#3498db', width=2),
                      marker=dict(size=4),
                      name='Position X'),
            row=1, col=2
        )
        
        # Y Position
        fig.add_trace(
            go.Scatter(x=df_valid['Frame'], y=df_valid['Y_center'],
                      mode='lines+markers',
                      line=dict(color='#e74c3c', width=2),
                      marker=dict(size=4),
                      name='Position Y'),
            row=2, col=1
        )
        
        # Radius evolution
        fig.add_trace(
            go.Scatter(x=df_valid['Frame'], y=df_valid['Radius'],
                      mode='lines+markers',
                      line=dict(color='#2ecc71', width=2),
                      marker=dict(size=4),
                      name='Rayon'),
            row=2, col=2
        )
        
        fig.update_layout(height=600, showlegend=False,
                         title_text="Analyse Complète de Détection")
        fig.update_yaxes(autorange="reversed", row=1, col=1)
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Detection statistics
        st.markdown("### 📊 Statistiques de Détection")
        
        stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
        
        with stat_col1:
            total_distance = np.sqrt(
                (df_valid['X_center'].iloc[-1] - df_valid['X_center'].iloc[0])**2 + 
                (df_valid['Y_center'].iloc[-1] - df_valid['Y_center'].iloc[0])**2
            )
            st.markdown(f"""
            <div class="metric-item">
                <div class="metric-value">{total_distance:.1f}</div>
                <div class="metric-label">Distance Totale (px)</div>
            </div>
            """, unsafe_allow_html=True)
        
        with stat_col2:
            if len(df_valid) > 1:
                dx = df_valid['X_center'].diff()
                dy = df_valid['Y_center'].diff()
                speed = np.sqrt(dx**2 + dy**2)
                avg_speed = speed.mean()
                st.markdown(f"""
                <div class="metric-item">
                    <div class="metric-value">{avg_speed:.2f}</div>
                    <div class="metric-label">Vitesse Moyenne (px/frame)</div>
                </div>
                """, unsafe_allow_html=True)
        
        with stat_col3:
            vertical_displacement = abs(df_valid['Y_center'].iloc[-1] - df_valid['Y_center'].iloc[0])
            st.markdown(f"""
            <div class="metric-item">
                <div class="metric-value">{vertical_displacement:.1f}</div>
                <div class="metric-label">Déplacement Vertical (px)</div>
            </div>
            """, unsafe_allow_html=True)
        
        with stat_col4:
            avg_radius = df_valid['Radius'].mean()
            radius_std = df_valid['Radius'].std()
            st.markdown(f"""
            <div class="metric-item">
                <div class="metric-value">{avg_radius:.1f} ± {radius_std:.1f}</div>
                <div class="metric-label">Rayon Moyen (px)</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("</div></div>", unsafe_allow_html=True)
        
    # ===== CODE 2: KRR ANALYSIS =====
    elif analysis_type == "📊 Code 2 : Analyse Krr":
        st.markdown("""
        <div class="analysis-results">
            <h2 class="results-header">📊 Code 2 : Analyse du Coefficient de Résistance au Roulement (Krr)</h2>
            <div class="results-content">
        """, unsafe_allow_html=True)
        
        # Sphere parameters section
        st.markdown("### 🔵 Paramètres de la Sphère")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            sphere_radius_mm = st.number_input("Rayon de la sphère (mm)", value=15.0, min_value=1.0, max_value=50.0)
            sphere_mass_g = st.number_input("Masse de la sphère (g)", value=10.0, min_value=0.1, max_value=1000.0)
            
        with col2:
            sphere_type_selection = st.selectbox("Type de sphère", ["Solide (j=2/5)", "Creuse (j=2/3)"])
            j_value = 2/5 if "Solide" in sphere_type_selection else 2/3
            
            # Density calculation
            volume_mm3 = (4/3) * np.pi * sphere_radius_mm**3
            volume_m3 = volume_mm3 * 1e-9
            mass_kg = sphere_mass_g * 1e-3
            density_kg_m3 = mass_kg / volume_m3
            
            st.markdown(f"""
            <div class="metric-item">
                <div class="metric-value">{density_kg_m3:.0f}</div>
                <div class="metric-label">Densité</div>
                <div class="metric-unit">kg/m³</div>
            </div>
            """, unsafe_allow_html=True)
            
        with col3:
            st.markdown(f"""
            <div class="metric-item">
                <div class="metric-value">{j_value:.3f}</div>
                <div class="metric-label">Facteur d'inertie j</div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class="metric-item">
                <div class="metric-value">{1/(1+j_value):.4f}</div>
                <div class="metric-label">Facteur (1+j)⁻¹</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Experimental parameters
        st.markdown("### 📐 Paramètres Expérimentaux")
        param_col1, param_col2, param_col3 = st.columns(3)
        
        with param_col1:
            fps = st.number_input("FPS de la caméra", value=250.0, min_value=1.0, max_value=1000.0)
            angle_deg = st.number_input("Angle d'inclinaison (°)", value=15.0, min_value=0.1, max_value=45.0)
            
        with param_col2:
            # Automatic calibration
            if len(df_valid) > 0:
                avg_radius_pixels = df_valid['Radius'].mean()
                auto_calibration = avg_radius_pixels / sphere_radius_mm
                
                st.markdown(f"""
                <div class="metric-item">
                    <div class="metric-value">{auto_calibration:.2f}</div>
                    <div class="metric-label">Calibration auto</div>
                    <div class="metric-unit">px/mm</div>
                </div>
                """, unsafe_allow_html=True)
                
                use_auto_cal = st.checkbox("Utiliser calibration automatique", value=True)
                if use_auto_cal:
                    pixels_per_mm = auto_calibration
                else:
                    pixels_per_mm = st.number_input("Calibration (px/mm)", value=auto_calibration, min_value=0.1)
            else:
                pixels_per_mm = st.number_input("Calibration (px/mm)", value=5.0, min_value=0.1)
                
        with param_col3:
            water_content_analysis = st.number_input("Teneur en eau (%)", value=water_content, min_value=0.0, max_value=100.0)
        
        # Kinematic calculations
        st.markdown("### 🧮 Calculs Cinématiques")
        
        if len(df_valid) > 10:
            # Unit conversion
            dt = 1 / fps
            
            # Positions in meters
            x_mm = df_valid['X_center'].values / pixels_per_mm
            y_mm = df_valid['Y_center'].values / pixels_per_mm
            x_m = x_mm / 1000
            y_m = y_mm / 1000
            
            # Time
            t = np.arange(len(df_valid)) * dt
            
            # Velocities
            vx = np.gradient(x_m, dt)
            vy = np.gradient(y_m, dt)
            v_magnitude = np.sqrt(vx**2 + vy**2)
            
            # Initial and final velocities
            n_avg = min(3, len(v_magnitude)//4)
            v0 = np.mean(v_magnitude[:n_avg])
            vf = np.mean(v_magnitude[-n_avg:])
            
            # Total distance
            distances = np.sqrt(np.diff(x_m)**2 + np.diff(y_m)**2)
            total_distance = np.sum(distances)
            
            # Calculate Krr coefficient
            g = 9.81
            if total_distance > 0:
                krr = (v0**2 - vf**2) / (2 * g * total_distance)
                
                # Results display matching your image layout
                st.markdown("### 📊 Résultats Krr")
                
                result_col1, result_col2, result_col3, result_col4 = st.columns(4)
                
                with result_col1:
                    st.markdown(f"""
                    <div class="metric-item">
                        <div class="metric-value">{v0*1000:.1f}</div>
                        <div class="metric-label">V₀ (vitesse initiale)</div>
                        <div class="metric-unit">mm/s</div>
                    </div>
                    """, unsafe_allow_html=True)
                    st.caption(f"{v0:.4f} m/s")
                
                with result_col2:
                    st.markdown(f"""
                    <div class="metric-item">
                        <div class="metric-value">{vf*1000:.1f}</div>
                        <div class="metric-label">Vf (vitesse finale)</div>
                        <div class="metric-unit">mm/s</div>
                    </div>
                    """, unsafe_allow_html=True)
                    st.caption(f"{vf:.4f} m/s")
                
                with result_col3:
                    st.markdown(f"""
                    <div class="metric-item">
                        <div class="metric-value">{total_distance*1000:.1f}</div>
                        <div class="metric-label">Distance totale</div>
                        <div class="metric-unit">mm</div>
                    </div>
                    """, unsafe_allow_html=True)
                    st.caption(f"{total_distance:.4f} m")
                
                with result_col4:
                    st.markdown(f"""
                    <div class="metric-item">
                        <div class="metric-value">{krr:.6f}</div>
                        <div class="metric-label"><strong>Coefficient Krr</strong></div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Validation with literature
                    if 0.03 <= krr <= 0.10:
                        st.markdown('<div class="status-success">✅ Cohérent avec Van Wal (2017)</div>', unsafe_allow_html=True)
                    elif krr < 0:
                        st.markdown('<div class="status-error">⚠️ Krr négatif - sphère accélère</div>', unsafe_allow_html=True)
                    else:
                        st.markdown('<div class="status-warning">⚠️ Différent de la littérature</div>', unsafe_allow_html=True)
                
                # Trajectory and velocity visualization (4 subplots as per your image)
                st.markdown("### 🎯 Trajectoire et Profil de Vitesse")
                
                fig_krr = make_subplots(
                    rows=2, cols=2,
                    subplot_titles=('Vitesse vs Temps', 'Accélération vs Temps', 
                                   'Trajectoire', 'Composantes de Vitesse'),
                    specs=[[{"secondary_y": False}, {"secondary_y": False}],
                           [{"secondary_y": False}, {"secondary_y": False}]]
                )
                
                # Velocity plot with reference lines
                fig_krr.add_trace(
                    go.Scatter(x=t, y=v_magnitude*1000, mode='lines', 
                              line=dict(color='blue', width=2), name='Vitesse'),
                    row=1, col=1
                )
                fig_krr.add_hline(y=v0*1000, line_dash="dash", line_color="green", row=1, col=1)
                fig_krr.add_hline(y=vf*1000, line_dash="dash", line_color="red", row=1, col=1)
                
                # Acceleration
                acceleration = np.gradient(v_magnitude, dt)
                fig_krr.add_trace(
                    go.Scatter(x=t, y=acceleration*1000, mode='lines',
                              line=dict(color='red', width=2), name='Accélération'),
                    row=1, col=2
                )
                
                # Trajectory
                fig_krr.add_trace(
                    go.Scatter(x=x_mm, y=y_mm, mode='markers+lines',
                              marker=dict(color=t, colorscale='viridis', size=4),
                              line=dict(width=2), name='Trajectoire'),
                    row=2, col=1
                )
                
                # Velocity components
                fig_krr.add_trace(
                    go.Scatter(x=t, y=np.abs(vx)*1000, mode='lines',
                              line=dict(color='blue', width=2), name='|Vx|'),
                    row=2, col=2
                )
                fig_krr.add_trace(
                    go.Scatter(x=t, y=vy*1000, mode='lines',
                              line=dict(color='red', width=2), name='Vy'),
                    row=2, col=2
                )
                
                fig_krr.update_layout(height=700, showlegend=False)
                fig_krr.update_xaxes(title_text="Temps (s)")
                fig_krr.update_yaxes(title_text="Vitesse (mm/s)", row=1, col=1)
                fig_krr.update_yaxes(title_text="Accélération (mm/s²)", row=1, col=2)
                fig_krr.update_yaxes(title_text="Y (mm)", row=2, col=1)
                fig_krr.update_yaxes(title_text="Vitesse (mm/s)", row=2, col=2)
                
                st.plotly_chart(fig_krr, use_container_width=True)
                
            else:
                st.error("❌ Distance parcourue nulle - impossible de calculer Krr")
        else:
            st.warning("⚠️ Pas assez de données valides pour l'analyse Krr")
            
        st.markdown("</div></div>", unsafe_allow_html=True)
        
    # ===== CODE 3: ADVANCED COMPLETE ANALYSIS =====
    elif analysis_type == "🔬 Code 3 : Analyse Complète Avancée":
        st.markdown("""
        <div class="analysis-results">
            <h2 class="results-header">🔬 Code 3 : Analyse Cinématique Avancée et Complète</h2>
            <div class="results-content">
        """, unsafe_allow_html=True)
        
        # Data verification section
        st.markdown("### 🔍 Vérification des Données")
        
        verify_col1, verify_col2, verify_col3 = st.columns(3)
        
        with verify_col1:
            st.markdown(f"""
            <div class="metric-item">
                <div class="metric-value">{len(df_valid)}</div>
                <div class="metric-label">Données valides</div>
                <div class="metric-unit">frames</div>
            </div>
            """, unsafe_allow_html=True)
            
            success_rate = len(df_valid)/len(st.session_state.current_df)*100
            st.markdown(f"""
            <div class="metric-item">
                <div class="metric-value">{success_rate:.1f}%</div>
                <div class="metric-label">Taux de succès</div>
            </div>
            """, unsafe_allow_html=True)
        
        with verify_col2:
            radius_range = df_valid['Radius'].max() - df_valid['Radius'].min()
            st.markdown(f"""
            <div class="metric-item">
                <div class="metric-value">{radius_range:.1f}</div>
                <div class="metric-label">Variation de rayon</div>
                <div class="metric-unit">px</div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class="metric-item">
                <div class="metric-value">{df_valid['Frame'].min()}</div>
                <div class="metric-label">Première détection</div>
                <div class="metric-unit">Frame</div>
            </div>
            """, unsafe_allow_html=True)
        
        with verify_col3:
            st.markdown(f"""
            <div class="metric-item">
                <div class="metric-value">{df_valid['Frame'].max()}</div>
                <div class="metric-label">Dernière détection</div>
                <div class="metric-unit">Frame</div>
            </div>
            """, unsafe_allow_html=True)
            
            duration_frames = df_valid['Frame'].max() - df_valid['Frame'].min()
            st.markdown(f"""
            <div class="metric-item">
                <div class="metric-value">{duration_frames}</div>
                <div class="metric-label">Durée de suivi</div>
                <div class="metric-unit">frames</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Advanced parameters
        st.markdown("### ⚙️ Paramètres d'Analyse Avancée")
        
        param_col1, param_col2, param_col3 = st.columns(3)
        
        with param_col1:
            st.markdown("**Paramètres Sphère**")
            mass_g = st.number_input("Masse (g)", value=10.0, min_value=0.1, key="adv_mass")
            radius_mm = st.number_input("Rayon (mm)", value=15.0, min_value=1.0, key="adv_radius")
            sphere_type_adv = st.selectbox("Type", ["Solide", "Creuse"], key="adv_type")
            j_factor = 2/5 if sphere_type_adv == "Solide" else 2/3
            
        with param_col2:
            st.markdown("**Paramètres Expérimentaux**")
            fps_adv = st.number_input("FPS", value=250.0, min_value=1.0, key="adv_fps")
            angle_deg_adv = st.number_input("Angle (°)", value=15.0, min_value=0.1, key="adv_angle")
            
            # Automatic calibration
            if len(df_valid) > 0:
                avg_radius_px = df_valid['Radius'].mean()
                auto_cal = avg_radius_px / radius_mm
                st.markdown(f"""
                <div class="metric-item">
                    <div class="metric-value">{auto_cal:.2f}</div>
                    <div class="metric-label">Calibration auto</div>
                    <div class="metric-unit">px/mm</div>
                </div>
                """, unsafe_allow_html=True)
                pixels_per_mm_adv = auto_cal
        
        with param_col3:
            st.markdown("**Filtrage des Données**")
            use_smoothing = st.checkbox("Lissage des données", value=True)
            smooth_window = st.slider("Fenêtre de lissage", 3, 11, 5, step=2)
            remove_outliers = st.checkbox("Supprimer les aberrants", value=True)
        
        # Launch analysis button
        if st.button("🚀 Lancer l'Analyse Complète"):
            with st.spinner("🧮 Calcul des métriques avancées en cours..."):
                metrics = calculate_advanced_metrics(df_valid, fps_adv, pixels_per_mm_adv, mass_g, angle_deg_adv)
            
            if metrics and metrics['krr'] is not None:
                
                # Main results display matching your image style
                st.markdown("### 📊 Résultats de l'Analyse Avancée")
                
                # Key metrics in grid layout
                adv_col1, adv_col2, adv_col3, adv_col4 = st.columns(4)
                
                with adv_col1:
                    st.markdown(f"""
                    <div class="metric-item">
                        <div class="metric-value">{metrics['krr']:.6f}</div>
                        <div class="metric-label">Krr Moyen</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown(f"""
                    <div class="metric-item">
                        <div class="metric-value">{metrics['energy_dissipated']*1000:.2f}</div>
                        <div class="metric-label">Énergie Dissipée</div>
                        <div class="metric-unit">mJ</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with adv_col2:
                    st.markdown(f"""
                    <div class="metric-item">
                        <div class="metric-value">{metrics['avg_power']*1000:.2f}</div>
                        <div class="metric-label">Puissance Moyenne</div>
                        <div class="metric-unit">mW</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with adv_col3:
                    st.markdown(f"""
                    <div class="metric-item">
                        <div class="metric-value">{metrics['max_velocity']*1000:.1f}</div>
                        <div class="metric-label">Vitesse Max</div>
                        <div class="metric-unit">mm/s</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown(f"""
                    <div class="metric-item">
                        <div class="metric-value">{metrics['max_acceleration']*1000:.1f}</div>
                        <div class="metric-label">Accél. Max</div>
                        <div class="metric-unit">mm/s²</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with adv_col4:
                    st.markdown(f"""
                    <div class="metric-item">
                        <div class="metric-value">{metrics['distance']*1000:.1f}</div>
                        <div class="metric-label">Distance Totale</div>
                        <div class="metric-unit">mm</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Advanced visualizations matching your image layout
                st.markdown("### 📈 Visualisations Avancées")
                
                fig_advanced = make_subplots(
                    rows=3, cols=2,
                    subplot_titles=('Vitesse Lissée vs Temps', 'Accélération vs Temps',
                                   'Énergies Cinétiques', 'Krr Instantané',
                                   'Puissance de Résistance', 'Forces'),
                    vertical_spacing=0.08,
                    specs=[[{"secondary_y": False}, {"secondary_y": False}],
                           [{"secondary_y": False}, {"secondary_y": False}],
                           [{"secondary_y": False}, {"secondary_y": False}]]
                )
                
                # Velocity plot
                fig_advanced.add_trace(
                    go.Scatter(x=metrics['time'], y=metrics['velocity']*1000, 
                             mode='lines', line=dict(color='blue', width=2), name='Vitesse'),
                    row=1, col=1
                )
                
                # Acceleration plot
                fig_advanced.add_trace(
                    go.Scatter(x=metrics['time'], y=metrics['acceleration']*1000,
                             mode='lines', line=dict(color='red', width=2), name='Accélération'),
                    row=1, col=2
                )
                
                # Energy plots
                E_trans = 0.5 * (mass_g/1000) * metrics['velocity']**2
                I = j_factor * (mass_g/1000) * (radius_mm/1000)**2
                omega = metrics['velocity'] / (radius_mm/1000)
                E_rot = 0.5 * I * omega**2
                E_total = E_trans + E_rot
                
                fig_advanced.add_trace(
                    go.Scatter(x=metrics['time'], y=E_trans*1000, mode='lines', 
                             line=dict(color='blue', width=2), name='Translation'),
                    row=2, col=1
                )
                fig_advanced.add_trace(
                    go.Scatter(x=metrics['time'], y=E_rot*1000, mode='lines', 
                             line=dict(color='red', width=2), name='Rotation'),
                    row=2, col=1
                )
                fig_advanced.add_trace(
                    go.Scatter(x=metrics['time'], y=E_total*1000, mode='lines', 
                             line=dict(color='black', width=3), name='Total'),
                    row=2, col=1
                )
                
                # Instantaneous Krr
                Krr_inst = np.abs(metrics['resistance_force']) / ((mass_g/1000) * 9.81)
                avg_krr = np.mean(Krr_inst)
                fig_advanced.add_trace(
                    go.Scatter(x=metrics['time'], y=Krr_inst, mode='lines', 
                             line=dict(color='purple', width=2), name='Krr'),
                    row=2, col=2
                )
                fig_advanced.add_hline(y=avg_krr, line_dash="dash", line_color="orange", row=2, col=2)
                
                # Power plot
                fig_advanced.add_trace(
                    go.Scatter(x=metrics['time'], y=metrics['power']*1000, mode='lines', 
                             line=dict(color='green', width=2), name='Puissance'),
                    row=3, col=1
                )
                
                # Forces plot
                F_gravity = (mass_g/1000) * 9.81 * np.sin(np.radians(angle_deg_adv))
                fig_advanced.add_trace(
                    go.Scatter(x=metrics['time'], y=metrics['resistance_force']*1000, mode='lines', 
                             line=dict(color='red', width=2), name='F_résistance'),
                    row=3, col=2
                )
                fig_advanced.add_hline(y=F_gravity*1000, line_dash="dash", line_color="blue", row=3, col=2)
                
                # Update layout
                fig_advanced.update_layout(height=900, showlegend=False)
                fig_advanced.update_xaxes(title_text="Temps (s)")
                fig_advanced.update_yaxes(title_text="Vitesse (mm/s)", row=1, col=1)
                fig_advanced.update_yaxes(title_text="Accélération (mm/s²)", row=1, col=2)
                fig_advanced.update_yaxes(title_text="Énergie (mJ)", row=2, col=1)
                fig_advanced.update_yaxes(title_text="Coefficient Krr", row=2, col=2)
                fig_advanced.update_yaxes(title_text="Puissance (mW)", row=3, col=1)
                fig_advanced.update_yaxes(title_text="Force (mN)", row=3, col=2)
                
                st.plotly_chart(fig_advanced, use_container_width=True)
                
                # Physical interpretation matching your image
                st.markdown("### 🧠 Analyse de Cohérence")
                
                coherence_col1, coherence_col2 = st.columns(2)
                
                with coherence_col1:
                    st.markdown("**Cohérence avec Van Wal (2017)**")
                    if 0.03 <= metrics['krr'] <= 0.10:
                        st.markdown(f'<div class="status-success">✅ Krr = {metrics["krr"]:.6f} cohérent avec littérature (0.05-0.07)</div>', unsafe_allow_html=True)
                    elif metrics['krr'] < 0:
                        st.markdown(f'<div class="status-error">❌ Krr négatif = {metrics["krr"]:.6f} - Sphère accélère</div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="status-warning">⚠️ Krr = {metrics["krr"]:.6f} différent de la littérature</div>', unsafe_allow_html=True)
                
                with coherence_col2:
                    st.markdown("**Bilan Énergétique**")
                    energy_ratio = (metrics['energy_dissipated'] / metrics['energy_initial']) * 100 if metrics['energy_initial'] > 0 else 0
                    st.markdown(f"""
                    <div class="metric-item">
                        <div class="metric-value">{energy_ratio:.1f}%</div>
                        <div class="metric-label">Énergie dissipée</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    if 10 <= energy_ratio <= 90:
                        st.markdown('<div class="status-success">✅ Dissipation énergétique cohérente</div>', unsafe_allow_html=True)
                    else:
                        st.markdown('<div class="status-warning">⚠️ Dissipation énergétique inhabituelle</div>', unsafe_allow_html=True)
                
                # Export options
                st.markdown("### 💾 Export des Résultats")
                
                export_col1, export_col2, export_col3 = st.columns(3)
                
                with export_col1:
                    basic_results = pd.DataFrame({
                        'Parametre': ['Krr', 'Vitesse_Max_mm/s', 'Distance_mm', 'Duree_s', 'Efficacite_Energie_%'],
                        'Valeur': [
                            metrics['krr'],
                            metrics['max_velocity']*1000,
                            metrics['distance']*1000,
                            metrics['duration'],
                            metrics['energy_efficiency']
                        ]
                    })
                    
                    csv_basic = basic_results.to_csv(index=False)
                    st.download_button(
                        label="📋 Résultats Principaux (CSV)",
                        data=csv_basic,
                        file_name="resultats_principaux.csv",
                        mime="text/csv"
                    )
                
                with export_col2:
                    detailed_data = pd.DataFrame({
                        'temps_s': metrics['time'],
                        'vitesse_mm_s': metrics['velocity']*1000,
                        'acceleration_mm_s2': metrics['acceleration']*1000,
                        'force_resistance_mN': metrics['resistance_force']*1000,
                        'puissance_mW': metrics['power']*1000,
                        'energie_cinetique_mJ': metrics['energy_kinetic']*1000
                    })
                    
                    csv_detailed = detailed_data.to_csv(index=False)
                    st.download_button(
                        label="📈 Données Temporelles (CSV)",
                        data=csv_detailed,
                        file_name="donnees_temporelles.csv",
                        mime="text/csv"
                    )
                
                with export_col3:
                    trajectory_data = df_valid.copy()
                    trajectory_data['temps_s'] = np.arange(len(trajectory_data)) / fps_adv
                    
                    csv_trajectory = trajectory_data.to_csv(index=False)
                    st.download_button(
                        label="🛤️ Données Trajectoire (CSV)",
                        data=csv_trajectory,
                        file_name="donnees_trajectoire.csv",
                        mime="text/csv"
                    )
                
            else:
                st.error("❌ Impossible de calculer les métriques - données insuffisantes")
        
        st.markdown("</div></div>", unsafe_allow_html=True)
    
    # ===== REPRODUCIBILITY ANALYSIS =====
    elif analysis_type == "🔄 Analyse de Reproductibilité":
        st.markdown("""
        <div class="analysis-results">
            <h2 class="results-header">🔄 Analyse de Reproductibilité et Détection d'Anomalies</h2>
            <div class="results-content">
        """, unsafe_allow_html=True)
        
        if not st.session_state.experiments:
            st.warning("⚠️ Aucune expérience disponible pour l'analyse de reproductibilité.")
            
            if st.button("📊 Charger des expériences d'exemple avec répétitions"):
                # Create sample experiments with repetitions
                conditions = [
                    (0, "Steel"), (10, "Steel"), (20, "Steel"),
                    (0, "Plastic"), (10, "Plastic"), (20, "Plastic")
                ]
                
                for water, material in conditions:
                    # Create 3 repetitions for each condition
                    for rep in range(1, 4):
                        # Add some variation for realism
                        variation = np.random.normal(0, 0.5)  # Small random variation
                        df_sample, metadata = create_sample_data_with_metadata(
                            f"{material}_W{water}%_Rep{rep}", 
                            water + variation, 
                            material
                        )
                        # Add noise to make some experiments anomalous
                        if rep == 2 and water == 10:  # Make second repetition of 10% water anomalous
                            # Modify data to create anomaly
                            df_sample.loc[df_sample['X_center'] != 0, 'X_center'] *= 0.8
                        
                        st.session_state.experiments[f"{material}_W{water}%_Rep{rep}"] = {
                            'data': df_sample,
                            'metadata': metadata
                        }
                st.success("✅ Expériences d'exemple avec répétitions chargées!")
                st.rerun()
        else:
            
            # Group experiments by conditions
            def group_experiments_by_conditions(experiments):
                groups = {}
                for exp_name, exp in experiments.items():
                    meta = exp['metadata']
                    # Create condition key (water content rounded to nearest 0.5, sphere type)
                    water_key = round(meta['water_content'] * 2) / 2  # Round to nearest 0.5
                    condition_key = f"{meta['sphere_type']}_W{water_key}%"
                    
                    if condition_key not in groups:
                        groups[condition_key] = []
                    
                    groups[condition_key].append({
                        'name': exp_name,
                        'data': exp['data'],
                        'metadata': meta
                    })
                
                return groups
            
            # Detect anomalies using statistical methods
            def detect_anomalies(group_data, threshold=2.0):
                """Detect anomalies in a group of experiments using Z-score"""
                if len(group_data) < 3:
                    return []
                
                # Calculate metrics for each experiment
                metrics_list = []
                for exp in group_data:
                    df_valid = exp['data'][(exp['data']['X_center'] != 0) & 
                                          (exp['data']['Y_center'] != 0) & 
                                          (exp['data']['Radius'] != 0)]
                    
                    if len(df_valid) > 10:
                        metrics = calculate_advanced_metrics(df_valid)
                        if metrics and metrics['krr'] is not None:
                            metrics_list.append({
                                'name': exp['name'],
                                'krr': metrics['krr'],
                                'max_velocity': metrics['max_velocity'],
                                'energy_efficiency': metrics['energy_efficiency'],
                                'trajectory_efficiency': metrics['trajectory_efficiency']
                            })
                
                if len(metrics_list) < 3:
                    return []
                
                # Calculate Z-scores for key metrics
                anomalies = []
                for metric_name in ['krr', 'max_velocity', 'energy_efficiency']:
                    values = [m[metric_name] for m in metrics_list if m[metric_name] is not None]
                    if len(values) >= 3:
                        mean_val = np.mean(values)
                        std_val = np.std(values)
                        
                        if std_val > 0:
                            for i, m in enumerate(metrics_list):
                                if m[metric_name] is not None:
                                    z_score = abs((m[metric_name] - mean_val) / std_val)
                                    if z_score > threshold:
                                        anomalies.append({
                                            'name': m['name'],
                                            'metric': metric_name,
                                            'value': m[metric_name],
                                            'z_score': z_score,
                                            'mean': mean_val,
                                            'std': std_val
                                        })
                
                return anomalies
            
            # Group experiments
            groups = group_experiments_by_conditions(st.session_state.experiments)
            
            st.markdown("### 📊 Groupement par Conditions Expérimentales")
            
            # Show groups summary
            group_summary = []
            for condition, experiments in groups.items():
                group_summary.append({
                    'Condition': condition,
                    'Nombre d\'expériences': len(experiments),
                    'Expériences': ', '.join([exp['name'] for exp in experiments])
                })
            
            group_df = pd.DataFrame(group_summary)
            st.dataframe(group_df, use_container_width=True)
            
            # Select condition for detailed analysis
            st.markdown("### 🎯 Sélection pour Analyse Détaillée")
            
            # Only show conditions with multiple experiments
            multi_exp_conditions = {k: v for k, v in groups.items() if len(v) >= 2}
            
            if not multi_exp_conditions:
                st.warning("⚠️ Aucune condition avec plusieurs expériences trouvée pour l'analyse de reproductibilité.")
            else:
                selected_condition = st.selectbox(
                    "Choisissez une condition à analyser:",
                    options=list(multi_exp_conditions.keys())
                )
                
                if selected_condition:
                    condition_experiments = multi_exp_conditions[selected_condition]
                    
                    st.markdown(f"### 🔍 Analyse de Reproductibilité - {selected_condition}")
                    
                    # Calculate metrics for all experiments in the group
                    experiment_metrics = []
                    valid_experiments = []
                    
                    for exp in condition_experiments:
                        df_valid = exp['data'][(exp['data']['X_center'] != 0) & 
                                              (exp['data']['Y_center'] != 0) & 
                                              (exp['data']['Radius'] != 0)]
                        
                        if len(df_valid) > 10:
                            metrics = calculate_advanced_metrics(df_valid)
                            if metrics and metrics['krr'] is not None:
                                experiment_metrics.append({
                                    'Expérience': exp['name'],
                                    'Krr': metrics['krr'],
                                    'Vitesse_Max': metrics['max_velocity'],
                                    'Efficacité_Énergie': metrics['energy_efficiency'],
                                    'Efficacité_Trajectoire': metrics['trajectory_efficiency'],
                                    'Distance': metrics['distance'],
                                    'Durée': metrics['duration']
                                })
                                valid_experiments.append(exp)
                    
                    if len(experiment_metrics) < 2:
                        st.error("❌ Pas assez d'expériences valides pour l'analyse de reproductibilité.")
                    else:
                        metrics_df = pd.DataFrame(experiment_metrics)
                        
                        # Detect anomalies
                        anomalies = detect_anomalies(condition_experiments, threshold=2.0)
                        
                        st.markdown("#### 🚨 Détection d'Anomalies")
                        
                        if anomalies:
                            st.markdown("**Anomalies détectées:**")
                            anomaly_df = pd.DataFrame(anomalies)
                            
                            for _, anomaly in anomaly_df.iterrows():
                                st.markdown(f"""
                                <div class="status-warning">
                                    ⚠️ <strong>{anomaly['name']}</strong> - {anomaly['metric']}: 
                                    {anomaly['value']:.6f} (Z-score: {anomaly['z_score']:.2f})
                                </div>
                                """, unsafe_allow_html=True)
                        else:
                            st.markdown('<div class="status-success">✅ Aucune anomalie détectée</div>', unsafe_allow_html=True)
                        
                        # Filter out anomalous experiments
                        anomalous_names = list(set([a['name'] for a in anomalies]))
                        filtered_metrics = metrics_df[~metrics_df['Expérience'].isin(anomalous_names)]
                        
                        if len(filtered_metrics) < 2:
                            st.warning("⚠️ Pas assez d'expériences valides après filtrage des anomalies.")
                        else:
                            st.markdown("#### 📊 Statistiques de Reproductibilité")
                            
                            # Calculate reproducibility statistics
                            repro_stats = []
                            numeric_cols = ['Krr', 'Vitesse_Max', 'Efficacité_Énergie', 'Efficacité_Trajectoire']
                            
                            for col in numeric_cols:
                                if col in filtered_metrics.columns:
                                    values = filtered_metrics[col].dropna()
                                    if len(values) > 0:
                                        mean_val = values.mean()
                                        std_val = values.std()
                                        cv = (std_val / mean_val * 100) if mean_val != 0 else 0
                                        
                                        repro_stats.append({
                                            'Paramètre': col,
                                            'Moyenne': mean_val,
                                            'Écart-type': std_val,
                                            'CV (%)': cv,
                                            'Min': values.min(),
                                            'Max': values.max(),
                                            'N': len(values)
                                        })
                            
                            repro_df = pd.DataFrame(repro_stats)
                            
                            # Display formatted statistics
                            st.markdown("##### 📈 Statistiques Descriptives")
                            
                            for _, stat in repro_df.iterrows():
                                col1, col2, col3, col4 = st.columns(4)
                                
                                with col1:
                                    st.markdown(f"""
                                    <div class="metric-item">
                                        <div class="metric-value">{stat['Moyenne']:.4f}</div>
                                        <div class="metric-label">{stat['Paramètre']} - Moyenne</div>
                                    </div>
                                    """, unsafe_allow_html=True)
                                
                                with col2:
                                    st.markdown(f"""
                                    <div class="metric-item">
                                        <div class="metric-value">{stat['Écart-type']:.4f}</div>
                                        <div class="metric-label">Écart-type</div>
                                    </div>
                                    """, unsafe_allow_html=True)
                                
                                with col3:
                                    cv_color = "red" if stat['CV (%)'] > 10 else "orange" if stat['CV (%)'] > 5 else "green"
                                    st.markdown(f"""
                                    <div class="metric-item">
                                        <div class="metric-value" style="color: {cv_color};">{stat['CV (%)']:.1f}%</div>
                                        <div class="metric-label">CV (Reproductibilité)</div>
                                    </div>
                                    """, unsafe_allow_html=True)
                                
                                with col4:
                                    st.markdown(f"""
                                    <div class="metric-item">
                                        <div class="metric-value">{stat['N']}</div>
                                        <div class="metric-label">Échantillons valides</div>
                                    </div>
                                    """, unsafe_allow_html=True)
                            
                            # Visualizations
                            st.markdown("#### 📈 Visualisations de Reproductibilité")
                            
                            # Create comparison plots
                            fig_repro = make_subplots(
                                rows=2, cols=2,
                                subplot_titles=('Coefficient Krr', 'Vitesse Maximale', 
                                               'Efficacité Énergétique', 'Efficacité Trajectoire'),
                                specs=[[{"secondary_y": False}, {"secondary_y": False}],
                                       [{"secondary_y": False}, {"secondary_y": False}]]
                            )
                            
                            # Plot each metric
                            plot_configs = [
                                ('Krr', 1, 1), ('Vitesse_Max', 1, 2), 
                                ('Efficacité_Énergie', 2, 1), ('Efficacité_Trajectoire', 2, 2)
                            ]
                            
                            for metric, row, col in plot_configs:
                                if metric in metrics_df.columns:
                                    # All experiments (including anomalous)
                                    fig_repro.add_trace(
                                        go.Scatter(
                                            x=metrics_df['Expérience'],
                                            y=metrics_df[metric],
                                            mode='markers',
                                            name=f'{metric} - Toutes',
                                            marker=dict(color='red', size=10, symbol='x'),
                                            showlegend=False
                                        ),
                                        row=row, col=col
                                    )
                                    
                                    # Filtered experiments (without anomalies)
                                    fig_repro.add_trace(
                                        go.Scatter(
                                            x=filtered_metrics['Expérience'],
                                            y=filtered_metrics[metric],
                                            mode='markers',
                                            name=f'{metric} - Filtrées',
                                            marker=dict(color='blue', size=12, symbol='circle'),
                                            showlegend=False
                                        ),
                                        row=row, col=col
                                    )
                                    
                                    # Add mean line
                                    if len(filtered_metrics) > 0:
                                        mean_val = filtered_metrics[metric].mean()
                                        fig_repro.add_hline(
                                            y=mean_val, 
                                            line_dash="dash", 
                                            line_color="green",
                                            row=row, col=col
                                        )
                                        
                                        # Add confidence bands (±1 std)
                                        std_val = filtered_metrics[metric].std()
                                        fig_repro.add_hline(
                                            y=mean_val + std_val, 
                                            line_dash="dot", 
                                            line_color="orange",
                                            row=row, col=col
                                        )
                                        fig_repro.add_hline(
                                            y=mean_val - std_val, 
                                            line_dash="dot", 
                                            line_color="orange",
                                            row=row, col=col
                                        )
                            
                            fig_repro.update_layout(height=600, showlegend=False)
                            fig_repro.update_xaxes(tickangle=45)
                            st.plotly_chart(fig_repro, use_container_width=True)
                            
                            # Time series comparison with averages
                            st.markdown("#### 📈 Comparaison des Séries Temporelles")
                            
                            # Calculate time series for all valid experiments
                            time_series_data = []
                            max_time_points = 0
                            
                            for exp in valid_experiments:
                                df_valid_exp = exp['data'][(exp['data']['X_center'] != 0) & 
                                                          (exp['data']['Y_center'] != 0) & 
                                                          (exp['data']['Radius'] != 0)]
                                
                                if len(df_valid_exp) > 10:
                                    metrics = calculate_advanced_metrics(df_valid_exp)
                                    if metrics and metrics['krr'] is not None:
                                        is_anomaly = exp['name'] in anomalous_names
                                        time_series_data.append({
                                            'name': exp['name'],
                                            'time': metrics['time'],
                                            'velocity': metrics['velocity'],
                                            'acceleration': metrics['acceleration'],
                                            'power': metrics['power'],
                                            'energy_kinetic': metrics['energy_kinetic'],
                                            'is_anomaly': is_anomaly
                                        })
                                        max_time_points = max(max_time_points, len(metrics['time']))
                            
                            if len(time_series_data) >= 2:
                                
                                # Create time series plots
                                fig_time_series = make_subplots(
                                    rows=2, cols=2,
                                    subplot_titles=('Vitesse vs Temps', 'Accélération vs Temps', 
                                                   'Puissance vs Temps', 'Énergie Cinétique vs Temps'),
                                    specs=[[{"secondary_y": False}, {"secondary_y": False}],
                                           [{"secondary_y": False}, {"secondary_y": False}]]
                                )
                                
                                # Plot configurations
                                plot_configs = [
                                    ('velocity', 'Vitesse (mm/s)', 1000, 1, 1),
                                    ('acceleration', 'Accélération (mm/s²)', 1000, 1, 2),
                                    ('power', 'Puissance (mW)', 1000, 2, 1),
                                    ('energy_kinetic', 'Énergie Cinétique (mJ)', 1000, 2, 2)
                                ]
                                
                                for metric, ylabel, scale_factor, row, col in plot_configs:
                                    # Plot individual experiments
                                    valid_series = []
                                    
                                    for ts_data in time_series_data:
                                        if metric in ts_data and ts_data[metric] is not None:
                                            color = 'red' if ts_data['is_anomaly'] else 'lightblue'
                                            opacity = 0.3 if ts_data['is_anomaly'] else 0.6
                                            line_width = 1 if ts_data['is_anomaly'] else 2
                                            
                                            fig_time_series.add_trace(
                                                go.Scatter(
                                                    x=ts_data['time'],
                                                    y=ts_data[metric] * scale_factor,
                                                    mode='lines',
                                                    name=ts_data['name'],
                                                    line=dict(color=color, width=line_width),
                                                    opacity=opacity,
                                                    showlegend=False
                                                ),
                                                row=row, col=col
                                            )
                                            
                                            # Collect valid series for averaging
                                            if not ts_data['is_anomaly']:
                                                valid_series.append(ts_data[metric])
                                    
                                    # Calculate and plot average curve
                                    if len(valid_series) >= 2:
                                        # Interpolate all series to common time grid
                                        max_time = max([max(ts_data['time']) for ts_data in time_series_data 
                                                       if not ts_data['is_anomaly']])
                                        min_time = min([min(ts_data['time']) for ts_data in time_series_data 
                                                       if not ts_data['is_anomaly']])
                                        
                                        # Create common time grid
                                        common_time = np.linspace(min_time, max_time, 100)
                                        interpolated_series = []
                                        
                                        for i, ts_data in enumerate(time_series_data):
                                            if not ts_data['is_anomaly'] and metric in ts_data:
                                                # Interpolate to common time grid
                                                interp_values = np.interp(common_time, ts_data['time'], ts_data[metric])
                                                interpolated_series.append(interp_values)
                                        
                                        if len(interpolated_series) >= 2:
                                            # Calculate mean and std
                                            mean_curve = np.mean(interpolated_series, axis=0)
                                            std_curve = np.std(interpolated_series, axis=0)
                                            
                                            # Plot mean curve
                                            fig_time_series.add_trace(
                                                go.Scatter(
                                                    x=common_time,
                                                    y=mean_curve * scale_factor,
                                                    mode='lines',
                                                    name=f'Moyenne {metric}',
                                                    line=dict(color='darkblue', width=4),
                                                    showlegend=False
                                                ),
                                                row=row, col=col
                                            )
                                            
                                            # Add confidence bands (±1 std)
                                            fig_time_series.add_trace(
                                                go.Scatter(
                                                    x=common_time,
                                                    y=(mean_curve + std_curve) * scale_factor,
                                                    mode='lines',
                                                    line=dict(color='darkblue', width=0),
                                                    showlegend=False,
                                                    hoverinfo='skip'
                                                ),
                                                row=row, col=col
                                            )
                                            
                                            fig_time_series.add_trace(
                                                go.Scatter(
                                                    x=common_time,
                                                    y=(mean_curve - std_curve) * scale_factor,
                                                    mode='lines',
                                                    line=dict(color='darkblue', width=0),
                                                    fill='tonexty',
                                                    fillcolor='rgba(0,0,139,0.2)',
                                                    showlegend=False,
                                                    hoverinfo='skip'
                                                ),
                                                row=row, col=col
                                            )
                                    
                                    # Update axes
                                    fig_time_series.update_yaxes(title_text=ylabel, row=row, col=col)
                                
                                fig_time_series.update_xaxes(title_text="Temps (s)")
                                fig_time_series.update_layout(
                                    height=700, 
                                    showlegend=False,
                                    title_text="Comparaison des Séries Temporelles avec Moyennes"
                                )
                                
                                st.plotly_chart(fig_time_series, use_container_width=True)
                                
                                # Add legend explanation
                                st.markdown("""
                                **Légende des courbes:**
                                - 🔴 **Lignes rouges fines** : Expériences anomales (exclues du calcul de moyenne)
                                - 🔵 **Lignes bleues claires** : Expériences valides
                                - 🔵 **Ligne bleue épaisse** : Moyenne des expériences valides
                                - 🔵 **Zone bleue transparente** : Bande de confiance (±1 écart-type)
                                """)
                                
                                # Statistical analysis of time series
                                st.markdown("#### 📊 Analyse Statistique des Séries Temporelles")
                                
                                # Calculate reproducibility metrics for time series
                                ts_stats = []
                                for metric, ylabel, scale_factor, row, col in plot_configs:
                                    valid_ts = [ts_data[metric] for ts_data in time_series_data 
                                               if not ts_data['is_anomaly'] and metric in ts_data]
                                    
                                    if len(valid_ts) >= 2:
                                        # Calculate point-wise statistics
                                        min_length = min([len(ts) for ts in valid_ts])
                                        truncated_series = [ts[:min_length] for ts in valid_ts]
                                        
                                        # Calculate mean CV across time points
                                        point_wise_cv = []
                                        for i in range(min_length):
                                            values = [ts[i] for ts in truncated_series]
                                            mean_val = np.mean(values)
                                            std_val = np.std(values)
                                            if mean_val != 0:
                                                cv = (std_val / mean_val) * 100
                                                point_wise_cv.append(cv)
                                        
                                        if point_wise_cv:
                                            avg_cv = np.mean(point_wise_cv)
                                            max_cv = np.max(point_wise_cv)
                                            
                                            ts_stats.append({
                                                'Métrique': metric.replace('_', ' ').title(),
                                                'CV Moyen (%)': avg_cv,
                                                'CV Max (%)': max_cv,
                                                'Expériences': len(valid_ts),
                                                'Points temporels': min_length
                                            })
                                
                                if ts_stats:
                                    ts_stats_df = pd.DataFrame(ts_stats)
                                    st.dataframe(ts_stats_df, use_container_width=True)
                                    
                                    # Overall time series reproducibility assessment
                                    avg_cv_all = np.mean([stat['CV Moyen (%)'] for stat in ts_stats])
                                    
                                    if avg_cv_all < 5:
                                        ts_grade = "Excellente"
                                        ts_color = "success"
                                    elif avg_cv_all < 10:
                                        ts_grade = "Bonne"
                                        ts_color = "success"
                                    elif avg_cv_all < 15:
                                        ts_grade = "Modérée"
                                        ts_color = "warning"
                                    else:
                                        ts_grade = "Faible"
                                        ts_color = "error"
                                    
                                    st.markdown(f"""
                                    <div class="status-{ts_color}">
                                        📈 <strong>Reproductibilité Temporelle: {ts_grade}</strong><br>
                                        • CV moyen des séries temporelles: {avg_cv_all:.1f}%<br>
                                        • Expériences valides: {len([ts for ts in time_series_data if not ts['is_anomaly']])}<br>
                                        • Cohérence temporelle: {"Élevée" if avg_cv_all < 10 else "Modérée" if avg_cv_all < 20 else "Faible"}
                                    </div>
                                    """, unsafe_allow_html=True)
                                
                            else:
                                st.warning("⚠️ Pas assez de séries temporelles valides pour la comparaison.")
                            
                            # Summary assessment
                            st.markdown("#### 🎯 Évaluation de la Reproductibilité")
                            
                            # Calculate overall reproducibility score
                            cv_values = [stat['CV (%)'] for stat in repro_stats if stat['CV (%)'] > 0]
                            if cv_values:
                                avg_cv = np.mean(cv_values)
                                
                                if avg_cv < 5:
                                    repro_grade = "Excellente"
                                    repro_color = "success"
                                elif avg_cv < 10:
                                    repro_grade = "Bonne"
                                    repro_color = "success"
                                elif avg_cv < 15:
                                    repro_grade = "Modérée"
                                    repro_color = "warning"
                                else:
                                    repro_grade = "Faible"
                                    repro_color = "error"
                                
                                st.markdown(f"""
                                <div class="status-{repro_color}">
                                    📊 <strong>Reproductibilité {repro_grade}</strong><br>
                                    • CV moyen: {avg_cv:.1f}%<br>
                                    • Expériences valides: {len(filtered_metrics)}/{len(metrics_df)}<br>
                                    • Anomalies détectées: {len(anomalous_names)}
                                </div>
                                """, unsafe_allow_html=True)
                            
                            # Export reproducibility report
                            st.markdown("#### 💾 Export Rapport de Reproductibilité")
                            
                            export_col1, export_col2 = st.columns(2)
                            
                            with export_col1:
                                # Summary report
                                repro_report = f"""
RAPPORT DE REPRODUCTIBILITÉ - {selected_condition}

=== RÉSUMÉ ===
Condition: {selected_condition}
Expériences totales: {len(metrics_df)}
Expériences valides: {len(filtered_metrics)}
Anomalies détectées: {len(anomalous_names)}

=== STATISTIQUES ===
"""
                                for _, stat in repro_df.iterrows():
                                    repro_report += f"""
{stat['Paramètre']}:
  Moyenne: {stat['Moyenne']:.6f}
  Écart-type: {stat['Écart-type']:.6f}
  CV: {stat['CV (%)']:.1f}%
  Gamme: {stat['Min']:.6f} - {stat['Max']:.6f}
"""
                                
                                if anomalies:
                                    repro_report += "\n=== ANOMALIES DÉTECTÉES ===\n"
                                    for anomaly in anomalies:
                                        repro_report += f"• {anomaly['name']} - {anomaly['metric']}: Z-score = {anomaly['z_score']:.2f}\n"
                                
                                st.download_button(
                                    label="📄 Rapport Reproductibilité (TXT)",
                                    data=repro_report,
                                    file_name=f"reproductibilite_{selected_condition}.txt",
                                    mime="text/plain"
                                )
                            
                            with export_col2:
                                # Detailed data
                                export_data = metrics_df.copy()
                                export_data['Anomalie'] = export_data['Expérience'].isin(anomalous_names)
                                
                                csv_repro = export_data.to_csv(index=False)
                                st.download_button(
                                    label="📊 Données Détaillées (CSV)",
                                    data=csv_repro,
                                    file_name=f"donnees_reproductibilite_{selected_condition}.csv",
                                    mime="text/csv"
                                )
        
        st.markdown("</div></div>", unsafe_allow_html=True)
    
    # ===== MULTI-EXPERIMENT COMPARISON =====
    elif analysis_type == "🔍 Comparaison Multi-Expériences":
        st.markdown("# 🔍 Comparaison Multi-Expériences")
        
        if not st.session_state.experiments:
            st.warning("⚠️ Aucune expérience disponible pour comparaison.")
            
            if st.button("📊 Charger des expériences d'exemple"):
                water_contents = [0, 5, 10, 15, 20]
                for w in water_contents:
                    df_sample, metadata = create_sample_data_with_metadata(f"Sample_W{w}%", w, "Steel")
                    st.session_state.experiments[f"Sample_W{w}%"] = {
                        'data': df_sample,
                        'metadata': metadata
                    }
                st.success("✅ Expériences d'exemple chargées!")
                st.rerun()
        else:
            # Show available experiments
            st.markdown("### 📋 Expériences Disponibles")
            
            exp_data = []
            for name, exp in st.session_state.experiments.items():
                meta = exp['metadata']
                exp_data.append({
                    'Expérience': name,
                    'Teneur en Eau (%)': meta['water_content'],
                    'Type de Sphère': meta['sphere_type'],
                    'Taux de Succès (%)': f"{meta['success_rate']:.1f}",
                    'Détections Valides': meta['valid_detections'],
                    'Date': meta['date']
                })
            
            exp_df = pd.DataFrame(exp_data)
            st.dataframe(exp_df, use_container_width=True)
            
            # Comparison interface
            selected_experiments = st.multiselect(
                "Choisissez les expériences pour comparaison:",
                options=list(st.session_state.experiments.keys()),
                default=list(st.session_state.experiments.keys())[:min(4, len(st.session_state.experiments))]
            )
            
            if len(selected_experiments) >= 2:
                comparison_data = []
                
                for exp_name in selected_experiments:
                    exp = st.session_state.experiments[exp_name]
                    df_exp = exp['data']
                    meta = exp['metadata']
                    df_exp_valid = df_exp[(df_exp['X_center'] != 0) & (df_exp['Y_center'] != 0) & (df_exp['Radius'] != 0)]
                    
                    metrics = calculate_advanced_metrics(df_exp_valid)
                    
                    comparison_data.append({
                        'Expérience': exp_name,
                        'Water_Content': meta['water_content'],
                        'Sphere_Type': meta['sphere_type'],
                        'Success_Rate': meta['success_rate'],
                        'Krr': metrics['krr'] if metrics else None,
                        'Max_Velocity': metrics['max_velocity'] if metrics else None,
                        'Energy_Efficiency': metrics['energy_efficiency'] if metrics else None,
                        'Trajectory_Efficiency': metrics['trajectory_efficiency'] if metrics else None,
                    })
                
                comp_df = pd.DataFrame(comparison_data)
                
                # Visualization
                st.markdown("### 📊 Analyses Comparatives")
                
                comp_col1, comp_col2 = st.columns(2)
                
                with comp_col1:
                    if comp_df['Krr'].notna().any():
                        fig_krr = px.scatter(comp_df, x='Water_Content', y='Krr', 
                                           color='Sphere_Type', size='Success_Rate',
                                           hover_data=['Expérience'],
                                           title="🔍 Krr vs Teneur en Eau")
                        st.plotly_chart(fig_krr, use_container_width=True)
                
                with comp_col2:
                    fig_success = px.bar(comp_df, x='Expérience', y='Success_Rate',
                                       color='Water_Content',
                                       title="📈 Taux de Succès de Détection")
                    fig_success.update_xaxes(tickangle=45)
                    st.plotly_chart(fig_success, use_container_width=True)
                
                # Comparison table
                st.markdown("### 📋 Tableau de Comparaison")
                
                display_comp = comp_df.copy()
                if 'Krr' in display_comp.columns:
                    display_comp['Krr'] = display_comp['Krr'].apply(lambda x: f"{x:.6f}" if pd.notna(x) else "N/A")
                
                st.dataframe(display_comp, use_container_width=True)
                
                # Export comparison
                csv_comparison = comp_df.to_csv(index=False)
                st.download_button(
                    label="📥 Télécharger comparaison (CSV)",
                    data=csv_comparison,
                    file_name="comparaison_experiences.csv",
                    mime="text/csv"
                )
            
            else:
                st.info("Veuillez sélectionner au moins 2 expériences pour la comparaison")
    
    # ===== PREDICTION MODULE =====
    elif analysis_type == "🎯 Module de Prédiction":
        st.markdown("# 🎯 Module de Prédiction")
        
        if not st.session_state.experiments:
            st.warning("⚠️ Aucune donnée expérimentale disponible pour les prédictions.")
            
            if st.button("📊 Charger des données d'exemple pour prédiction"):
                water_contents = [0, 5, 10, 15, 20, 25]
                for w in water_contents:
                    df_sample, metadata = create_sample_data_with_metadata(f"Sample_W{w}%", w, "Steel")
                    st.session_state.experiments[f"Sample_W{w}%"] = {
                        'data': df_sample,
                        'metadata': metadata
                    }
                st.success("✅ Données d'exemple chargées!")
                st.rerun()
        else:
            # Build simple prediction models
            all_data = []
            for exp_name, exp in st.session_state.experiments.items():
                df_exp = exp['data']
                meta = exp['metadata']
                df_exp_valid = df_exp[(df_exp['X_center'] != 0) & (df_exp['Y_center'] != 0) & (df_exp['Radius'] != 0)]
                
                metrics = calculate_advanced_metrics(df_exp_valid)
                if metrics and metrics['krr'] is not None:
                    all_data.append({
                        'water_content': meta['water_content'],
                        'krr': metrics['krr'],
                        'energy_efficiency': metrics['energy_efficiency']
                    })
            
            if len(all_data) >= 3:
                model_df = pd.DataFrame(all_data)
                
                # Simple linear regression for Krr
                water_vals = model_df['water_content'].values
                krr_vals = model_df['krr'].values
                
                # Fit polynomial
                coeffs = np.polyfit(water_vals, krr_vals, 1)
                
                st.markdown("### 🔮 Prédictions")
                
                pred_water = st.slider("Teneur en Eau pour Prédiction (%)", 0.0, 30.0, 10.0, 0.5)
                
                # Make prediction
                predicted_krr = np.polyval(coeffs, pred_water)
                
                st.markdown(f"""
                <div class="metric-item">
                    <div class="metric-value">{predicted_krr:.6f}</div>
                    <div class="metric-label">Krr Prédit</div>
                </div>
                """, unsafe_allow_html=True)
                
                # Show prediction plot
                water_range = np.linspace(0, 30, 100)
                krr_pred_range = np.polyval(coeffs, water_range)
                
                fig_pred = go.Figure()
                fig_pred.add_trace(go.Scatter(x=water_vals, y=krr_vals, mode='markers',
                                            name='Données expérimentales', marker=dict(color='red', size=10)))
                fig_pred.add_trace(go.Scatter(x=water_range, y=krr_pred_range, mode='lines',
                                            name='Prédiction', line=dict(color='blue', width=2)))
                fig_pred.add_trace(go.Scatter(x=[pred_water], y=[predicted_krr], mode='markers',
                                            name='Prédiction actuelle', marker=dict(color='green', size=15, symbol='star')))
                
                fig_pred.update_layout(
                    title="📈 Modèle de Prédiction Krr",
                    xaxis_title="Teneur en Eau (%)",
                    yaxis_title="Coefficient Krr"
                )
                
                st.plotly_chart(fig_pred, use_container_width=True)
                
                # Model equation
                a, b = coeffs
                st.markdown(f"**Équation du modèle:** Krr = {a:.6f} × W + {b:.6f}")
                
            else:
                st.error("❌ Données insuffisantes pour construire un modèle prédictif. Besoin d'au moins 3 expériences.")
    
    # ===== AUTO-GENERATED REPORT =====
    elif analysis_type == "📄 Rapport Auto-Généré":
        st.markdown("# 📄 Rapport d'Analyse Auto-Généré")
        
        if not st.session_state.experiments:
            st.warning("⚠️ Aucune donnée expérimentale disponible pour la génération de rapport.")
            
            if st.button("📊 Charger des données d'exemple pour rapport"):
                conditions = [
                    (0, "Steel"), (5, "Steel"), (10, "Steel"), (15, "Steel"), (20, "Steel"),
                    (10, "Plastic"), (15, "Plastic")
                ]
                
                for water, material in conditions:
                    df_sample, metadata = create_sample_data_with_metadata(f"{material}_W{water}%", water, material)
                    st.session_state.experiments[f"{material}_W{water}%"] = {
                        'data': df_sample,
                        'metadata': metadata
                    }
                st.success("✅ Données d'exemple complètes chargées!")
                st.rerun()
        else:
            # Generate comprehensive report
            with st.spinner("🔄 Génération du rapport..."):
                
                total_experiments = len(st.session_state.experiments)
                water_contents = [exp['metadata']['water_content'] for exp in st.session_state.experiments.values()]
                success_rates = [exp['metadata']['success_rate'] for exp in st.session_state.experiments.values()]
                
                # Calculate metrics for all experiments
                all_metrics = []
                for exp_name, exp in st.session_state.experiments.items():
                    df_exp = exp['data']
                    meta = exp['metadata']
                    df_exp_valid = df_exp[(df_exp['X_center'] != 0) & (df_exp['Y_center'] != 0) & (df_exp['Radius'] != 0)]
                    
                    metrics = calculate_advanced_metrics(df_exp_valid)
                    if metrics:
                        all_metrics.append({
                            'exp_name': exp_name,
                            'water_content': meta['water_content'],
                            'sphere_type': meta['sphere_type'],
                            **metrics
                        })
            
            # Display report summary
            st.markdown("### 📋 Résumé du Rapport")
            
            report_col1, report_col2, report_col3, report_col4 = st.columns(4)
            
            with report_col1:
                st.markdown(f"""
                <div class="metric-item">
                    <div class="metric-value">{total_experiments}</div>
                    <div class="metric-label">Expériences Totales</div>
                </div>
                """, unsafe_allow_html=True)
            
            with report_col2:
                water_range = max(water_contents) - min(water_contents)
                st.markdown(f"""
                <div class="metric-item">
                    <div class="metric-value">{water_range:.1f}%</div>
                    <div class="metric-label">Gamme d'Humidité</div>
                </div>
                """, unsafe_allow_html=True)
            
            with report_col3:
                st.markdown(f"""
                <div class="metric-item">
                    <div class="metric-value">{np.mean(success_rates):.1f}%</div>
                    <div class="metric-label">Succès Moyen</div>
                </div>
                """, unsafe_allow_html=True)
            
            with report_col4:
                st.markdown(f"""
                <div class="metric-item">
                    <div class="metric-value">{len(all_metrics)}</div>
                    <div class="metric-label">Analyses Valides</div>
                </div>
                """, unsafe_allow_html=True)
            
            # Generate text report
            current_time = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            
            report_content = f"""
# 📊 RAPPORT D'ANALYSE AUTOMATIQUE
## Résistance au Roulement des Sphères sur Matériau Granulaire Humide

**Généré le:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Nombre d'expériences:** {total_experiments}

## 🎯 RÉSUMÉ EXÉCUTIF

### Principales Découvertes:
• **Expériences analysées**: {total_experiments}
• **Gamme d'humidité testée**: {min(water_contents):.1f}% - {max(water_contents):.1f}%
• **Succès de détection moyen**: {np.mean(success_rates):.1f}%

### Métriques Clés:
"""
            
            if all_metrics:
                all_krr = [m['krr'] for m in all_metrics if m['krr'] is not None]
                if all_krr:
                    report_content += f"• **Gamme coefficient Krr**: {min(all_krr):.6f} - {max(all_krr):.6f}\n"
                
                all_efficiency = [m['energy_efficiency'] for m in all_metrics if m['energy_efficiency'] is not None]
                if all_efficiency:
                    report_content += f"• **Gamme efficacité énergétique**: {min(all_efficiency):.1f}% - {max(all_efficiency):.1f}%\n"
            
            report_content += """
## 🔧 RECOMMANDATIONS D'INGÉNIERIE

• **Applications industrielles**: Maintenir la teneur en eau ±2% de l'optimum
• **Transport longue distance**: Utiliser une teneur en eau plus faible pour l'efficacité
• **Applications de précision**: Surveiller l'humidité en continu

## 📊 QUALITÉ DES DONNÉES
"""
            
            avg_success = np.mean(success_rates)
            if avg_success >= 80:
                report_content += "✅ **Excellente qualité de détection** - Résultats très fiables\n"
            elif avg_success >= 70:
                report_content += "✅ **Bonne qualité de détection** - Résultats fiables\n"
            else:
                report_content += "⚠️ **Qualité de détection modérée** - Considérer l'amélioration du setup\n"
            
            report_content += """
## 📞 CONTACT & MÉTHODOLOGIE

**Institution de Recherche**: Department of Cosmic Earth Science, Graduate School of Science, Osaka University
**Domaine de Recherche**: Mécanique Granulaire
**Innovation**: Première étude systématique des effets d'humidité sur la résistance au roulement

**Méthodologie**: Suivi de sphères par vision par ordinateur avec analyse cinématique
**Détection**: Soustraction d'arrière-plan avec transformées de Hough circulaires
**Analyse**: Calcul Krr utilisant les principes de conservation d'énergie
"""
            
            # Export options
            export_col1, export_col2 = st.columns(2)
            
            with export_col1:
                st.download_button(
                    label="📥 Télécharger le rapport (TXT)",
                    data=report_content,
                    file_name=f"rapport_analyse_{current_time}.txt",
                    mime="text/plain"
                )
            
            with export_col2:
                if all_metrics:
                    comprehensive_data = []
                    for metric in all_metrics:
                        comprehensive_data.append({
                            'Experiment': metric['exp_name'],
                            'Water_Content_percent': metric['water_content'],
                            'Sphere_Type': metric['sphere_type'],
                            'Krr_coefficient': metric['krr'],
                            'Max_Velocity_mm_s': metric['max_velocity'] * 1000 if metric['max_velocity'] else None,
                            'Energy_Efficiency_percent': metric['energy_efficiency'],
                            'Distance_traveled_mm': metric['distance'] * 1000 if metric['distance'] else None
                        })
                    
                    comprehensive_df = pd.DataFrame(comprehensive_data)
                    csv_comprehensive = comprehensive_df.to_csv(index=False)
                    
                    st.download_button(
                        label="📊 Données complètes (CSV)",
                        data=csv_comprehensive,
                        file_name=f"donnees_completes_{current_time}.csv",
                        mime="text/csv"
                    )
            
            # Quality assessment
            st.markdown("### 🎯 Évaluation de la Qualité")
            
            quality_col1, quality_col2 = st.columns(2)
            
            with quality_col1:
                st.markdown("#### Qualité des Données")
                
                if avg_success >= 80:
                    st.markdown('<div class="status-success">✅ Excellente qualité de détection</div>', unsafe_allow_html=True)
                    quality_score = "A+"
                elif avg_success >= 70:
                    st.markdown('<div class="status-success">✅ Bonne qualité de détection</div>', unsafe_allow_html=True)
                    quality_score = "A"
                elif avg_success >= 60:
                    st.markdown('<div class="status-warning">⚠️ Qualité de détection modérée</div>', unsafe_allow_html=True)
                    quality_score = "B"
                else:
                    st.markdown('<div class="status-error">❌ Qualité de détection faible</div>', unsafe_allow_html=True)
                    quality_score = "C"
                
                st.markdown(f"""
                <div class="metric-item">
                    <div class="metric-value">{quality_score}</div>
                    <div class="metric-label">Score de Qualité</div>
                </div>
                """, unsafe_allow_html=True)
            
            with quality_col2:
                st.markdown("#### Cohérence des Résultats")
                
                if all_metrics:
                    valid_krr = [m['krr'] for m in all_metrics if m['krr'] is not None and 0.01 <= m['krr'] <= 0.20]
                    coherent_count = len(valid_krr)
                    total_count = len([m for m in all_metrics if m['krr'] is not None])
                    
                    if total_count > 0:
                        coherence_rate = (coherent_count / total_count) * 100
                        
                        if coherence_rate >= 80:
                            st.markdown('<div class="status-success">✅ Résultats très cohérents</div>', unsafe_allow_html=True)
                        elif coherence_rate >= 60:
                            st.markdown('<div class="status-warning">⚠️ Résultats modérément cohérents</div>', unsafe_allow_html=True)
                        else:
                            st.markdown('<div class="status-error">❌ Résultats peu cohérents</div>', unsafe_allow_html=True)
                        
                        st.markdown(f"""
                        <div class="metric-item">
                            <div class="metric-value">{coherence_rate:.1f}%</div>
                            <div class="metric-label">Taux de Cohérence</div>
                        </div>
                        """, unsafe_allow_html=True)
            
            # Recommendations for improvement
            st.markdown("### 💡 Recommandations d'Amélioration")
            
            recommendations = []
            
            if total_experiments < 5:
                recommendations.append("📊 **Augmenter le nombre d'expériences** - Collecter au moins 5-8 expériences")
            
            if max(water_contents) - min(water_contents) < 15:
                recommendations.append("💧 **Élargir la gamme d'humidité** - Tester une gamme plus large")
            
            if avg_success < 75:
                recommendations.append("🔧 **Améliorer la qualité de détection** - Optimiser les paramètres")
            
            sphere_types = set([exp['metadata']['sphere_type'] for exp in st.session_state.experiments.values()])
            if len(sphere_types) < 2:
                recommendations.append("⚪ **Tester différents matériaux** - Inclure plusieurs types de sphères")
            
            if recommendations:
                for rec in recommendations:
                    st.markdown(rec)
            else:
                st.markdown('<div class="status-success">✅ Configuration expérimentale excellente!</div>', unsafe_allow_html=True)

else:
    # No data loaded message
    st.markdown("""
    <div class="section-card">
        <h2>🚀 Pour commencer</h2>
        <p>Téléchargez vos données expérimentales ou utilisez les données d'exemple pour explorer la plateforme.</p>
        
        <h3>📋 Format de fichier attendu:</h3>
        <ul>
            <li><strong>Frame</strong>: Numéro d'image</li>
            <li><strong>X_center</strong>: Position X du centre de la sphère</li>
            <li><strong>Y_center</strong>: Position Y du centre de la sphère</li>
            <li><strong>Radius</strong>: Rayon détecté de la sphère</li>
        </ul>
        
        <h3>🔧 Analyses Disponibles:</h3>
        <ul>
            <li><strong>Code 1</strong>: Visualisation de trajectoire et qualité de détection</li>
            <li><strong>Code 2</strong>: Analyse Krr avec comparaison littérature</li>
            <li><strong>Code 3</strong>: Analyse complète avec métriques avancées</li>
            <li><strong>Comparaison</strong>: Analyse multi-expériences</li>
            <li><strong>Prédiction</strong>: Modèles prédictifs</li>
            <li><strong>Rapport</strong>: Génération automatique de rapports</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# Sidebar - Show saved experiments
if st.session_state.experiments:
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 💾 Expériences Sauvegardées")
    for exp_name, exp in st.session_state.experiments.items():
        meta = exp['metadata']
        st.sidebar.markdown(f"""
        <div style="background: #f8f9fa; padding: 0.5rem; border-radius: 4px; margin: 0.5rem 0; font-size: 0.9rem;">
            <strong>{exp_name}</strong><br>
            💧 Eau: {meta['water_content']}%<br>
            ⚪ Type: {meta['sphere_type']}<br>
            ✅ Succès: {meta['success_rate']:.1f}%
        </div>
        """, unsafe_allow_html=True)
    
    # Quick management
    st.sidebar.markdown("#### 🗂️ Gestion Rapide")
    if st.sidebar.button("🧹 Effacer toutes les expériences"):
        st.session_state.experiments = {}
        st.success("Toutes les expériences effacées!")
        st.rerun()
else:
    st.sidebar.markdown("---")
    st.sidebar.info("💡 Chargez des expériences depuis la section de données pour les comparer et analyser.")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #6c757d; padding: 1rem;">
    <h4>🎓 Plateforme d'Analyse de Résistance au Roulement des Sphères</h4>
    <p>Développée pour l'analyse de la résistance au roulement des sphères sur matériau granulaire humide</p>
    <p><strong>Institution:</strong> Department of Cosmic Earth Science, Graduate School of Science, Osaka University</p>
    <p><strong>Innovation:</strong> Première étude de l'effet de l'humidité sur la résistance au roulement</p>
</div>
""", unsafe_allow_html=True)
