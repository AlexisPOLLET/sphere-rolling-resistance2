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
    .friction-card {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
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

# ==================== NEW: FRICTION ANALYSIS FUNCTIONS ====================

def calculate_friction_coefficients(df_valid, sphere_mass_g=10.0, angle_deg=15.0, fps=250.0, pixels_per_mm=5.0):
    """
    Calcule les différents coefficients de friction à partir des données de trajectoire
    """
    if len(df_valid) < 10:
        return None
    
    # Paramètres physiques
    mass_kg = sphere_mass_g / 1000
    angle_rad = np.radians(angle_deg)
    g = 9.81
    dt = 1 / fps
    
    # Conversion des positions en mètres
    x_m = df_valid['X_center'].values / pixels_per_mm / 1000
    y_m = df_valid['Y_center'].values / pixels_per_mm / 1000
    
    # Calcul des vitesses et accélérations
    vx = np.gradient(x_m, dt)
    vy = np.gradient(y_m, dt)
    v_magnitude = np.sqrt(vx**2 + vy**2)
    
    # Accélération
    acceleration = np.gradient(v_magnitude, dt)
    
    # Forces
    F_gravity_component = mass_kg * g * np.sin(angle_rad)  # Composante motrice
    F_normal = mass_kg * g * np.cos(angle_rad)  # Force normale
    F_resistance = mass_kg * np.abs(acceleration)  # Force de résistance totale
    
    # Coefficients de friction
    mu_kinetic = F_resistance / F_normal  # Coefficient de friction cinétique
    mu_rolling = mu_kinetic - np.tan(angle_rad)  # Coefficient de roulement pur
    
    # Vitesses initiales et finales
    n_avg = min(3, len(v_magnitude)//4)
    v0 = np.mean(v_magnitude[:n_avg]) if len(v_magnitude) >= n_avg else v_magnitude[0]
    vf = np.mean(v_magnitude[-n_avg:]) if len(v_magnitude) >= n_avg else v_magnitude[-1]
    
    # Distance totale
    distances = np.sqrt(np.diff(x_m)**2 + np.diff(y_m)**2)
    total_distance = np.sum(distances)
    
    # Krr traditionnel
    krr = (v0**2 - vf**2) / (2 * g * total_distance) if total_distance > 0 and v0 > vf else None
    
    # Analyse énergétique
    E_kinetic_initial = 0.5 * mass_kg * v0**2
    E_kinetic_final = 0.5 * mass_kg * vf**2
    E_potential_lost = mass_kg * g * total_distance * np.sin(angle_rad)
    E_dissipated = E_kinetic_initial - E_kinetic_final - E_potential_lost
    
    # Coefficient de friction énergétique
    mu_energetic = E_dissipated / (F_normal * total_distance) if total_distance > 0 else None
    
    return {
        # Coefficients de friction
        'mu_kinetic_avg': np.mean(mu_kinetic),
        'mu_kinetic_max': np.max(mu_kinetic),
        'mu_kinetic_min': np.min(mu_kinetic),
        'mu_rolling_avg': np.mean(mu_rolling),
        'mu_energetic': mu_energetic,
        
        # Krr et paramètres traditionnels
        'krr': krr,
        'v0': v0,
        'vf': vf,
        'total_distance': total_distance,
        
        # Forces
        'F_resistance_avg': np.mean(F_resistance),
        'F_normal': F_normal,
        'F_gravity_component': F_gravity_component,
        
        # Énergies
        'E_dissipated': E_dissipated,
        'energy_efficiency': (E_kinetic_final / E_kinetic_initial * 100) if E_kinetic_initial > 0 else 0,
        
        # Séries temporelles pour graphiques
        'time': np.arange(len(df_valid)) * dt,
        'mu_kinetic_series': mu_kinetic,
        'mu_rolling_series': mu_rolling,
        'velocity': v_magnitude,
        'acceleration': acceleration,
        'F_resistance_series': F_resistance
    }

def analyze_trace_friction(depth_mm, width_mm, length_mm, sphere_radius_mm, sphere_mass_g):
    """
    Analyse la friction à partir des dimensions de la trace laissée
    """
    # Volume déplacé
    volume_displaced = depth_mm * width_mm * length_mm  # mm³
    
    # Ratio de pénétration δ/R
    penetration_ratio = depth_mm / sphere_radius_mm
    
    # Énergie de déformation (approximation)
    deformation_energy = volume_displaced * penetration_ratio
    
    # Coefficient de friction apparent basé sur la géométrie
    friction_index = (depth_mm * width_mm) / (sphere_radius_mm ** 2)
    
    return {
        'volume_displaced_mm3': volume_displaced,
        'penetration_ratio': penetration_ratio,
        'deformation_energy_index': deformation_energy,
        'friction_geometric_index': friction_index,
        'width_to_diameter_ratio': width_mm / (2 * sphere_radius_mm)
    }

# ==================== MAIN APPLICATION ====================

# Header
st.markdown("""
<div class="main-header">
    <h1>📊 Plateforme d'Analyse de Résistance au Roulement des Sphères</h1>
    <p>Suite d'analyse complète pour la recherche en mécanique granulaire - Université d'Osaka</p>
    <p><strong>🔥 NOUVEAU:</strong> Analyse de friction grain-sphère intégrée!</p>
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
        "🔬 Code 3 : Analyse Complète + Friction",  # NOUVEAU : Titre modifié
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
if (st.session_state.current_df is not None and 
    st.session_state.current_df_valid is not None):
    
    df = st.session_state.current_df
    df_valid = st.session_state.current_df_valid
    
    if len(df_valid) > 0:
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
    else:
        st.warning("⚠️ Aucune détection valide dans les données chargées.")

# ==================== ANALYSIS SECTIONS ====================

if (st.session_state.current_df_valid is not None and 
    len(st.session_state.current_df_valid) > 0):
    
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
        
    # ===== CODE 3: ADVANCED COMPLETE ANALYSIS + FRICTION =====
    elif analysis_type == "🔬 Code 3 : Analyse Complète + Friction":
        st.markdown("""
        <div class="analysis-results">
            <h2 class="results-header">🔬 Code 3 : Analyse Cinématique Avancée + Analyse de Friction</h2>
            <div class="results-content">
                <p><strong>🔥 NOUVEAU :</strong> Analyse de friction grain-sphère intégrée dans l'analyse complète!</p>
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
        st.markdown("### ⚙️ Paramètres d'Analyse Avancée + Friction")
        
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
        
        # === NOUVELLE SECTION TRACE AVEC SESSION STATE ===
        st.markdown("### 🛤️ Paramètres de la Trace (Optionnel)")
        st.markdown("*Si vous avez mesuré la trace laissée par la sphère, entrez les dimensions :*")
        
        # Initialiser les valeurs de trace dans session_state si elles n'existent pas
        if 'trace_depth' not in st.session_state:
            st.session_state.trace_depth = 0.0
        if 'trace_width' not in st.session_state:
            st.session_state.trace_width = 0.0
        if 'trace_length' not in st.session_state:
            st.session_state.trace_length = 0.0
        if 'enable_trace_analysis' not in st.session_state:
            st.session_state.enable_trace_analysis = False
        
        # Checkbox pour activer l'analyse de trace
        enable_trace = st.checkbox("🔬 Activer l'analyse de trace", 
                                  value=st.session_state.enable_trace_analysis,
                                  help="Cochez cette case pour analyser les dimensions de la trace physique")
        
        # Mettre à jour le session state
        if enable_trace != st.session_state.enable_trace_analysis:
            st.session_state.enable_trace_analysis = enable_trace
        
        # Afficher les champs de trace seulement si activé
        if st.session_state.enable_trace_analysis:
            trace_col1, trace_col2, trace_col3 = st.columns(3)
            
            with trace_col1:
                depth_mm = st.number_input("Profondeur δ (mm)", 
                                         value=st.session_state.trace_depth, 
                                         min_value=0.0, 
                                         max_value=50.0,
                                         step=0.1,
                                         key="trace_depth_input",
                                         help="Profondeur de pénétration de la sphère dans le substrat")
                # Mettre à jour le session state
                if depth_mm != st.session_state.trace_depth:
                    st.session_state.trace_depth = depth_mm
                    
            with trace_col2:
                width_mm = st.number_input("Largeur (mm)", 
                                         value=st.session_state.trace_width, 
                                         min_value=0.0, 
                                         max_value=100.0,
                                         step=0.1,
                                         key="trace_width_input",
                                         help="Largeur de la trace laissée par la sphère")
                # Mettre à jour le session state
                if width_mm != st.session_state.trace_width:
                    st.session_state.trace_width = width_mm
                    
            with trace_col3:
                length_mm = st.number_input("Longueur (mm)", 
                                          value=st.session_state.trace_length, 
                                          min_value=0.0, 
                                          max_value=1000.0,
                                          step=0.1,
                                          key="trace_length_input",
                                          help="Longueur de la trace visible")
                # Mettre à jour le session state
                if length_mm != st.session_state.trace_length:
                    st.session_state.trace_length = length_mm
            
            # Affichage des valeurs actuelles
            if st.session_state.trace_depth > 0 or st.session_state.trace_width > 0 or st.session_state.trace_length > 0:
                st.markdown("#### 📏 Dimensions de Trace Enregistrées")
                trace_info_col1, trace_info_col2, trace_info_col3 = st.columns(3)
                
                with trace_info_col1:
                    st.markdown(f"""
                    <div class="metric-item">
                        <div class="metric-value">{st.session_state.trace_depth:.2f}</div>
                        <div class="metric-label">Profondeur δ</div>
                        <div class="metric-unit">mm</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                with trace_info_col2:
                    st.markdown(f"""
                    <div class="metric-item">
                        <div class="metric-value">{st.session_state.trace_width:.2f}</div>
                        <div class="metric-label">Largeur</div>
                        <div class="metric-unit">mm</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                with trace_info_col3:
                    st.markdown(f"""
                    <div class="metric-item">
                        <div class="metric-value">{st.session_state.trace_length:.2f}</div>
                        <div class="metric-label">Longueur</div>
                        <div class="metric-unit">mm</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Bouton pour effacer les valeurs de trace
                if st.button("🧹 Effacer les paramètres de trace"):
                    st.session_state.trace_depth = 0.0
                    st.session_state.trace_width = 0.0
                    st.session_state.trace_length = 0.0
                    st.session_state.enable_trace_analysis = False
                    st.success("Paramètres de trace effacés!")
                    st.rerun()
        
        # Launch analysis button
        if st.button("🚀 Lancer l'Analyse Complète + Friction"):
            with st.spinner("🧮 Calcul des métriques avancées et analyse de friction..."):
                metrics = calculate_advanced_metrics(df_valid, fps_adv, pixels_per_mm_adv, mass_g, angle_deg_adv)
            
            if metrics and metrics['krr'] is not None:
                
                # === FRICTION ANALYSIS SECTION ===
                st.markdown("### 🔥 Analyse de Friction Grain-Sphère")
                
                friction_results = calculate_friction_coefficients(
                    df_valid, 
                    sphere_mass_g=mass_g,
                    angle_deg=angle_deg_adv,
                    fps=fps_adv,
                    pixels_per_mm=pixels_per_mm_adv
                )
                
                if friction_results:
                    # Display friction results in nice cards
                    st.markdown("#### 📊 Coefficients de Friction Calculés")
                    
                    friction_col1, friction_col2, friction_col3, friction_col4 = st.columns(4)
                    
                    with friction_col1:
                        st.markdown(f"""
                        <div class="friction-card">
                            <h4>🔥 μ Cinétique</h4>
                            <h2>{friction_results['mu_kinetic_avg']:.4f}</h2>
                            <p>Friction grain-sphère directe</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                    with friction_col2:
                        st.markdown(f"""
                        <div class="friction-card">
                            <h4>🎯 μ Roulement</h4>
                            <h2>{friction_results['mu_rolling_avg']:.4f}</h2>
                            <p>Résistance pure au roulement</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                    with friction_col3:
                        mu_energetic_val = friction_results['mu_energetic'] if friction_results['mu_energetic'] else 0
                        st.markdown(f"""
                        <div class="friction-card">
                            <h4>⚡ μ Énergétique</h4>
                            <h2>{mu_energetic_val:.4f}</h2>
                            <p>Basé sur dissipation d'énergie</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                    with friction_col4:
                        krr_val = friction_results['krr'] if friction_results['krr'] else 0
                        st.markdown(f"""
                        <div class="friction-card">
                            <h4>📊 Krr Référence</h4>
                            <h2>{krr_val:.6f}</h2>
                            <p>Coefficient traditionnel</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Force analysis
                    st.markdown("#### ⚖️ Analyse des Forces")
                    
                    force_col1, force_col2, force_col3 = st.columns(3)
                    
                    with force_col1:
                        st.markdown(f"""
                        <div class="metric-item">
                            <div class="metric-value">{friction_results['F_normal']*1000:.2f}</div>
                            <div class="metric-label">Force Normale</div>
                            <div class="metric-unit">mN</div>
                        </div>
                        """, unsafe_allow_html=True)
                        st.caption("Composante perpendiculaire")
                        
                    with force_col2:
                        st.markdown(f"""
                        <div class="metric-item">
                            <div class="metric-value">{friction_results['F_resistance_avg']*1000:.2f}</div>
                            <div class="metric-label">Force Résistance Moyenne</div>
                            <div class="metric-unit">mN</div>
                        </div>
                        """, unsafe_allow_html=True)
                        st.caption("Opposition au mouvement")
                        
                    with force_col3:
                        st.markdown(f"""
                        <div class="metric-item">
                            <div class="metric-value">{friction_results['F_gravity_component']*1000:.2f}</div>
                            <div class="metric-label">Force Gravité (Composante)</div>
                            <div class="metric-unit">mN</div>
                        </div>
                        """, unsafe_allow_html=True)
                        st.caption("Force motrice")
                    
                    # Trace analysis if data is available
                    if (st.session_state.enable_trace_analysis and 
                        st.session_state.trace_depth > 0 and 
                        st.session_state.trace_width > 0 and 
                        st.session_state.trace_length > 0):
                        
                        st.markdown("#### 🛤️ Analyse de la Trace Mesurée")
                        
                        trace_results = analyze_trace_friction(
                            st.session_state.trace_depth, 
                            st.session_state.trace_width, 
                            st.session_state.trace_length, 
                            radius_mm, 
                            mass_g
                        )
                        
                        st.markdown("##### 📏 Résultats de l'Analyse de Trace")
                        
                        trace_res_col1, trace_res_col2, trace_res_col3 = st.columns(3)
                        
                        with trace_res_col1:
                            st.markdown(f"""
                            <div class="metric-item">
                                <div class="metric-value">{trace_results['penetration_ratio']:.3f}</div>
                                <div class="metric-label">Ratio Pénétration δ/R</div>
                            </div>
                            """, unsafe_allow_html=True)
                            st.markdown(f"""
                            <div class="metric-item">
                                <div class="metric-value">{trace_results['volume_displaced_mm3']:.1f}</div>
                                <div class="metric-label">Volume Déplacé</div>
                                <div class="metric-unit">mm³</div>
                            </div>
                            """, unsafe_allow_html=True)
                            
                        with trace_res_col2:
                            st.markdown(f"""
                            <div class="metric-item">
                                <div class="metric-value">{trace_results['friction_geometric_index']:.3f}</div>
                                <div class="metric-label">Indice Friction Géométrique</div>
                            </div>
                            """, unsafe_allow_html=True)
                            st.markdown(f"""
                            <div class="metric-item">
                                <div class="metric-value">{trace_results['deformation_energy_index']:.1f}</div>
                                <div class="metric-label">Indice Énergie Déformation</div>
                            </div>
                            """, unsafe_allow_html=True)
                            
                        with trace_res_col3:
                            st.markdown(f"""
                            <div class="metric-item">
                                <div class="metric-value">{trace_results['width_to_diameter_ratio']:.3f}</div>
                                <div class="metric-label">Ratio Largeur/Diamètre</div>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Literature comparison
                            if trace_results['penetration_ratio'] < 0.1:
                                st.markdown('<div class="status-success">✅ Faible pénétration (sol dur)</div>', unsafe_allow_html=True)
                            elif trace_results['penetration_ratio'] < 0.3:
                                st.markdown('<div class="status-success">ℹ️ Pénétration modérée</div>', unsafe_allow_html=True)
                            else:
                                st.markdown('<div class="status-warning">⚠️ Forte pénétration (sol mou)</div>', unsafe_allow_html=True)
                        
                        # Comparison with Darbois Texier (2018)
                        st.markdown("##### 🔬 Comparaison avec la Littérature")
                        
                        # Assume granular density ~1500 kg/m³, sphere density from mass and volume
                        sphere_volume = (4/3) * np.pi * (radius_mm/1000)**3
                        sphere_density = (mass_g/1000) / sphere_volume
                        granular_density = 1500  # kg/m³, typical for sand
                        density_ratio = sphere_density / granular_density
                        
                        # Darbois Texier relationship: δ/R ∝ (ρs/ρg)^0.75
                        expected_penetration = 0.1 * (density_ratio**0.75)  # Rough estimation
                        
                        col_lit1, col_lit2 = st.columns(2)
                        
                        with col_lit1:
                            st.markdown(f"""
                            <div class="metric-item">
                                <div class="metric-value">{trace_results['penetration_ratio']:.3f}</div>
                                <div class="metric-label">δ/R Mesuré</div>
                            </div>
                            """, unsafe_allow_html=True)
                            st.markdown(f"""
                            <div class="metric-item">
                                <div class="metric-value">{sphere_density:.0f}</div>
                                <div class="metric-label">Densité Sphère</div>
                                <div class="metric-unit">kg/m³</div>
                            </div>
                            """, unsafe_allow_html=True)
                            
                        with col_lit2:
                            st.markdown(f"""
                            <div class="metric-item">
                                <div class="metric-value">{expected_penetration:.3f}</div>
                                <div class="metric-label">δ/R Attendu (Darbois Texier)</div>
                            </div>
                            """, unsafe_allow_html=True)
                            st.markdown(f"""
                            <div class="metric-item">
                                <div class="metric-value">{density_ratio:.2f}</div>
                                <div class="metric-label">Ratio ρs/ρg</div>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    # Main results display
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
                    
                    # Advanced visualizations
                    st.markdown("### 📈 Visualisations Avancées + Analyse de Friction")
                    
                    fig_advanced = make_subplots(
                        rows=4, cols=2,
                        subplot_titles=('Vitesse Lissée vs Temps', 'Accélération vs Temps',
                                       'Énergies Cinétiques', 'Krr Instantané',
                                       'Puissance de Résistance', 'Forces',
                                       'Coefficients de Friction μ', 'Corrélation Force-Vitesse'),
                        vertical_spacing=0.06,
                        specs=[[{"secondary_y": False}, {"secondary_y": False}],
                               [{"secondary_y": False}, {"secondary_y": False}],
                               [{"secondary_y": False}, {"secondary_y": False}],
                               [{"secondary_y": False}, {"secondary_y": False}]]
                    )
                    
                    # 1. Velocity plot
                    fig_advanced.add_trace(
                        go.Scatter(x=metrics['time'], y=metrics['velocity']*1000, 
                                 mode='lines', line=dict(color='blue', width=2), name='Vitesse'),
                        row=1, col=1
                    )
                    
                    # 2. Acceleration plot
                    fig_advanced.add_trace(
                        go.Scatter(x=metrics['time'], y=metrics['acceleration']*1000,
                                 mode='lines', line=dict(color='red', width=2), name='Accélération'),
                        row=1, col=2
                    )
                    
                    # 3. Energy plots
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
                    
                    # 4. Instantaneous Krr
                    Krr_inst = np.abs(metrics['resistance_force']) / ((mass_g/1000) * 9.81)
                    avg_krr = np.mean(Krr_inst)
                    fig_advanced.add_trace(
                        go.Scatter(x=metrics['time'], y=Krr_inst, mode='lines', 
                                 line=dict(color='purple', width=2), name='Krr'),
                        row=2, col=2
                    )
                    fig_advanced.add_hline(y=avg_krr, line_dash="dash", line_color="orange", row=2, col=2)
                    
                    # 5. Power plot
                    fig_advanced.add_trace(
                        go.Scatter(x=metrics['time'], y=metrics['power']*1000, mode='lines', 
                                 line=dict(color='green', width=2), name='Puissance'),
                        row=3, col=1
                    )
                    
                    # 6. Forces plot
                    F_gravity = (mass_g/1000) * 9.81 * np.sin(np.radians(angle_deg_adv))
                    fig_advanced.add_trace(
                        go.Scatter(x=metrics['time'], y=metrics['resistance_force']*1000, mode='lines', 
                                 line=dict(color='red', width=2), name='F_résistance'),
                        row=3, col=2
                    )
                    fig_advanced.add_hline(y=F_gravity*1000, line_dash="dash", line_color="blue", row=3, col=2)
                    
                    # 7. Friction coefficients
                    fig_advanced.add_trace(
                        go.Scatter(x=friction_results['time'], y=friction_results['mu_kinetic_series'], 
                                  mode='lines', name='μ cinétique',
                                  line=dict(color='darkred', width=2)),
                        row=4, col=1
                    )
                    fig_advanced.add_trace(
                        go.Scatter(x=friction_results['time'], y=friction_results['mu_rolling_series'], 
                                  mode='lines', name='μ roulement',
                                  line=dict(color='orange', width=2)),
                        row=4, col=1
                    )
                    fig_advanced.add_hline(y=friction_results['mu_kinetic_avg'], 
                                          line_dash="dash", line_color="darkred", row=4, col=1)
                    
                    # 8. Force vs Velocity correlation
                    fig_advanced.add_trace(
                        go.Scatter(x=friction_results['velocity']*1000, 
                                  y=friction_results['F_resistance_series']*1000,
                                  mode='markers', name='F vs v',
                                  marker=dict(color='darkblue', size=4, opacity=0.7)),
                        row=4, col=2
                    )
                    
                    # Update layout
                    fig_advanced.update_layout(height=1200, showlegend=False)
                    fig_advanced.update_xaxes(title_text="Temps (s)")
                    fig_advanced.update_yaxes(title_text="Vitesse (mm/s)", row=1, col=1)
                    fig_advanced.update_yaxes(title_text="Accélération (mm/s²)", row=1, col=2)
                    fig_advanced.update_yaxes(title_text="Énergie (mJ)", row=2, col=1)
                    fig_advanced.update_yaxes(title_text="Coefficient Krr", row=2, col=2)
                    fig_advanced.update_yaxes(title_text="Puissance (mW)", row=3, col=1)
                    fig_advanced.update_yaxes(title_text="Force (mN)", row=3, col=2)
                    fig_advanced.update_yaxes(title_text="Coefficient de Friction", row=4, col=1)
                    fig_advanced.update_yaxes(title_text="Force Résistance (mN)", row=4, col=2)
                    
                    st.plotly_chart(fig_advanced, use_container_width=True)
                    
                    # Physical interpretation with friction insights
                    st.markdown("### 🧠 Interprétation Physique + Friction")
                    
                    coherence_col1, coherence_col2 = st.columns(2)
                    
                    with coherence_col1:
                        st.markdown("**Cohérence avec Van Wal (2017)**")
                        if 0.03 <= metrics['krr'] <= 0.10:
                            st.markdown(f'<div class="status-success">✅ Krr = {metrics["krr"]:.6f} cohérent avec littérature (0.05-0.07)</div>', unsafe_allow_html=True)
                        elif metrics['krr'] < 0:
                            st.markdown(f'<div class="status-error">❌ Krr négatif = {metrics["krr"]:.6f} - Sphère accélère</div>', unsafe_allow_html=True)
                        else:
                            st.markdown(f'<div class="status-warning">⚠️ Krr = {metrics["krr"]:.6f} différent de la littérature</div>', unsafe_allow_html=True)
                        
                        # Friction analysis
                        st.markdown("**Analyse de Friction**")
                        if friction_results['mu_kinetic_avg'] > 0.1:
                            st.markdown('<div class="status-warning">⚠️ Friction élevée - substrat très résistant</div>', unsafe_allow_html=True)
                        elif friction_results['mu_kinetic_avg'] > 0.05:
                            st.markdown('<div class="status-success">ℹ️ Friction modérée - cohérent avec attentes</div>', unsafe_allow_html=True)
                        else:
                            st.markdown('<div class="status-success">✅ Friction faible - roulement efficace</div>', unsafe_allow_html=True)
                    
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
                        
                        # Effect of humidity
                        st.markdown("**Effet de l'Humidité Attendu**")
                        st.markdown(f"""
                        **Teneur en eau actuelle :** {water_content}%
                        
                        **Votre résultat μ = {friction_results['mu_kinetic_avg']:.4f}**
                        
                        **Effets physiques attendus :**
                        - 💧 **0-5%** : Friction minimale (grains secs)
                        - 🌊 **5-15%** : Augmentation (ponts capillaires)
                        - 🌧️ **15-25%** : Maximum puis diminution (lubrification)
                        """)
                    
                    # Export enhanced options
                    st.markdown("### 💾 Export des Résultats Complets + Friction")
                    
                    export_col1, export_col2, export_col3 = st.columns(3)
                    
                    with export_col1:
                        # Enhanced basic results with friction
                        enhanced_results = pd.DataFrame({
                            'Parametre': ['Krr', 'Vitesse_Max_mm/s', 'Distance_mm', 'Duree_s', 'Efficacite_Energie_%',
                                         'μ_Cinétique', 'μ_Roulement', 'μ_Énergétique', 'Force_Normale_mN'],
                            'Valeur': [
                                metrics['krr'],
                                metrics['max_velocity']*1000,
                                metrics['distance']*1000,
                                metrics['duration'],
                                metrics['energy_efficiency'],
                                friction_results['mu_kinetic_avg'],
                                friction_results['mu_rolling_avg'],
                                friction_results['mu_energetic'],
                                friction_results['F_normal']*1000
                            ]
                        })
                        
                        csv_enhanced = enhanced_results.to_csv(index=False)
                        st.download_button(
                            label="📋 Résultats + Friction (CSV)",
                            data=csv_enhanced,
                            file_name="resultats_avec_friction.csv",
                            mime="text/csv"
                        )
                    
                    with export_col2:
                        # Enhanced detailed data with friction
                        detailed_data = pd.DataFrame({
                            'temps_s': metrics['time'],
                            'vitesse_mm_s': metrics['velocity']*1000,
                            'acceleration_mm_s2': metrics['acceleration']*1000,
                            'force_resistance_mN': metrics['resistance_force']*1000,
                            'puissance_mW': metrics['power']*1000,
                            'energie_cinetique_mJ': metrics['energy_kinetic']*1000
                        })
                        
                        # Add friction data
                        friction_data = pd.DataFrame({
                            'mu_kinetic': friction_results['mu_kinetic_series'],
                            'mu_rolling': friction_results['mu_rolling_series'],
                            'F_normal_mN': [friction_results['F_normal']*1000] * len(friction_results['mu_kinetic_series'])
                        })
                        
                        # Ensure same length
                        min_len = min(len(detailed_data), len(friction_data))
                        detailed_data = detailed_data.iloc[:min_len]
                        friction_data = friction_data.iloc[:min_len]
                        detailed_data = pd.concat([detailed_data, friction_data], axis=1)
                        
                        csv_detailed_friction = detailed_data.to_csv(index=False)
                        st.download_button(
                            label="📈 Données Temporelles + Friction (CSV)",
                            data=csv_detailed_friction,
                            file_name="donnees_temporelles_friction.csv",
                            mime="text/csv"
                        )
                    
                    with export_col3:
                        # Comprehensive friction report
                        friction_report = f"""
RAPPORT COMPLET D'ANALYSE DE FRICTION

=== PARAMÈTRES EXPÉRIMENTAUX ===
Teneur en eau: {water_content}%
Type de sphère: {sphere_type}
Masse: {mass_g}g
Rayon: {radius_mm}mm
Angle: {angle_deg_adv}°

=== COEFFICIENTS DE FRICTION ===
μ cinétique moyen: {friction_results['mu_kinetic_avg']:.4f}
μ roulement moyen: {friction_results['mu_rolling_avg']:.4f}
μ énergétique: {friction_results['mu_energetic']:.4f}
Krr: {friction_results['krr']:.6f}

=== FORCES ET ÉNERGIES ===
Force normale: {friction_results['F_normal']*1000:.2f} mN
Force résistance moyenne: {friction_results['F_resistance_avg']*1000:.2f} mN
Énergie dissipée: {friction_results['E_dissipated']*1000:.2f} mJ
Efficacité énergétique: {friction_results['energy_efficiency']:.1f}%

=== CINÉMATIQUE ===
Vitesse initiale: {friction_results['v0']*1000:.2f} mm/s
Vitesse finale: {friction_results['vf']*1000:.2f} mm/s
Distance totale: {friction_results['total_distance']*1000:.2f} mm
                        """
                        
                        st.download_button(
                            label="📄 Rapport Friction Complet (TXT)",
                            data=friction_report,
                            file_name="rapport_friction_complet.txt",
                            mime="text/plain"
                        )
                    
                else:
                    st.error("❌ Impossible de calculer les coefficients de friction")
            else:
                st.error("❌ Impossible de calculer les métriques - données insuffisantes")
            
        st.markdown("</div></div>", unsafe_allow_html=True)
