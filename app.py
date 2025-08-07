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
    """Nettoie robustement les données en éliminant les artefacts de début/fin"""
    if len(df_valid) < 20:
        return df_valid, {"error": "Pas assez de données"}
    
    # Conversion en unités physiques
    dt = 1 / fps
    x_m = df_valid['X_center'].values / pixels_per_mm / 1000
    y_m = df_valid['Y_center'].values / pixels_per_mm / 1000
    
    # Calcul des vitesses
    vx = np.gradient(x_m, dt)
    vy = np.gradient(y_m, dt)
    v_magnitude = np.sqrt(vx**2 + vy**2)
    
    # Calcul des accélérations
    acceleration = np.abs(np.gradient(v_magnitude, dt))
    
    # Méthode 1: Éliminer les N premiers et derniers points (brutal mais efficace)
    n_remove = max(5, len(df_valid) // 10)  # Enlever au moins 5 points ou 10% des données
    
    # Méthode 2: Détecter les zones de vitesse stable
    v_smooth = np.convolve(v_magnitude, np.ones(5)/5, mode='same')  # Lissage
    v_median = np.median(v_smooth)
    v_threshold = v_median * 0.3  # Seuil à 30% de la vitesse médiane
    
    # Trouver la zone stable
    stable_mask = v_smooth > v_threshold
    
    # Trouver les indices de début et fin de zone stable
    stable_indices = np.where(stable_mask)[0]
    
    if len(stable_indices) > 10:
        start_idx = stable_indices[0] + 3  # Ajouter une marge
        end_idx = stable_indices[-1] - 3   # Ajouter une marge
    else:
        # Fallback: utiliser la méthode brutale
        start_idx = n_remove
        end_idx = len(df_valid) - n_remove
    
    # S'assurer qu'on a encore assez de données
    if end_idx - start_idx < 10:
        # Utiliser une approche plus conservatrice
        start_idx = len(df_valid) // 5
        end_idx = len(df_valid) - len(df_valid) // 5
    
    # Découper les données
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
    
    # NETTOYAGE ROBUSTE DES DONNÉES
    df_clean, cleaning_info = clean_data_robust(df_valid, fps, pixels_per_mm)
    
    if "error" in cleaning_info:
        return None
    
    # Convert to real units avec données nettoyées
    dt = 1 / fps
    mass_kg = sphere_mass_g / 1000
    angle_rad = np.radians(angle_deg)
    g = 9.81
    
    x_m = df_clean['X_center'].values / pixels_per_mm / 1000
    y_m = df_clean['Y_center'].values / pixels_per_mm / 1000
    
    # Time array
    t = np.arange(len(df_clean)) * dt
    
    # Calculate velocities and accelerations avec données nettoyées
    vx = np.gradient(x_m, dt)
    vy = np.gradient(y_m, dt)
    v_magnitude = np.sqrt(vx**2 + vy**2)
    
    # Accelerations avec lissage pour éviter le bruit
    acceleration_raw = np.gradient(v_magnitude, dt)
    # Lissage de l'accélération pour éliminer les pics
    acceleration = np.convolve(acceleration_raw, np.ones(3)/3, mode='same')
    
    # Forces avec accélération lissée
    F_resistance = mass_kg * np.abs(acceleration)
    
    # Energies
    E_kinetic = 0.5 * mass_kg * v_magnitude**2
    E_initial = E_kinetic[0] if len(E_kinetic) > 0 else 0
    E_final = E_kinetic[-1] if len(E_kinetic) > 0 else 0
    E_dissipated = E_initial - E_final
    
    # Power
    P_resistance = F_resistance * v_magnitude
    
    # Basic Krr calculation avec données nettoyées
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
        'cleaning_info': cleaning_info  # Ajouter les infos de nettoyage
    }

def calculate_friction_coefficients(df_valid, sphere_mass_g=10.0, angle_deg=15.0, fps=250.0, pixels_per_mm=5.0):
    """Calculate friction coefficients from trajectory data"""
    if len(df_valid) < 10:
        return None
    
    # NETTOYAGE ROBUSTE DES DONNÉES POUR FRICTION AUSSI
    df_clean, cleaning_info = clean_data_robust(df_valid, fps, pixels_per_mm)
    
    if "error" in cleaning_info:
        return None
    
    # Physical parameters
    mass_kg = sphere_mass_g / 1000
    angle_rad = np.radians(angle_deg)
    g = 9.81
    dt = 1 / fps
    
    # Convert positions to meters avec données nettoyées
    x_m = df_clean['X_center'].values / pixels_per_mm / 1000
    y_m = df_clean['Y_center'].values / pixels_per_mm / 1000
    
    # Calculate velocities and accelerations avec données nettoyées
    vx = np.gradient(x_m, dt)
    vy = np.gradient(y_m, dt)
    v_magnitude = np.sqrt(vx**2 + vy**2)
    
    # Acceleration avec lissage
    acceleration_raw = np.gradient(v_magnitude, dt)
    acceleration = np.convolve(acceleration_raw, np.ones(3)/3, mode='same')
    
    # Forces
    F_gravity_component = mass_kg * g * np.sin(angle_rad)
    F_normal = mass_kg * g * np.cos(angle_rad)
    F_resistance = mass_kg * np.abs(acceleration)
    
    # Friction coefficients avec lissage
    mu_kinetic_raw = F_resistance / F_normal
    mu_kinetic = np.convolve(mu_kinetic_raw, np.ones(5)/5, mode='same')  # Lissage supplémentaire
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
        'cleaning_info': cleaning_info  # Ajouter les infos de nettoyage
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

# Sidebar
st.sidebar.markdown("### 📋 Navigation")

# Analysis type selection
analysis_type = st.sidebar.selectbox(
    "Sélectionnez le type d'analyse :",
    [
        "📈 Code 1 : Visualisation de Trajectoire",
        "📊 Code 2 : Analyse Krr", 
        "🔬 Code 3 : Analyse Complète + Friction",
        "🔍 Comparaison Multi-Expériences"
    ]
)

# ==================== DATA LOADING SECTION ====================

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
        "Choisissez votre fichier CSV", 
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

# ==================== ANALYSIS SECTIONS ====================

if (st.session_state.current_df_valid is not None and 
    len(st.session_state.current_df_valid) > 0):
    
    df_valid = st.session_state.current_df_valid
    
    # Quick overview
    st.markdown("## 📊 Aperçu des Données")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-item">
            <div class="metric-value">{len(st.session_state.current_df)}</div>
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
        success_rate = len(df_valid) / len(st.session_state.current_df) * 100 if len(st.session_state.current_df) > 0 else 0
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
    
    # ===== CODE 3: ADVANCED COMPLETE ANALYSIS + FRICTION =====
    if analysis_type == "🔬 Code 3 : Analyse Complète + Friction":
        st.markdown("## 🔬 Code 3 : Analyse Cinématique Avancée + Analyse de Friction")
        st.markdown("**🔥 NOUVEAU :** Analyse de friction grain-sphère intégrée!")
        
        # Parameters
        st.markdown("### ⚙️ Paramètres d'Analyse")
        
        param_col1, param_col2, param_col3 = st.columns(3)
        
        with param_col1:
            st.markdown("**Paramètres Sphère**")
            mass_g = st.number_input("Masse (g)", value=10.0, min_value=0.1)
            radius_mm = st.number_input("Rayon (mm)", value=15.0, min_value=1.0)
            
        with param_col2:
            st.markdown("**Paramètres Expérimentaux**")
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
                    <div class="metric-label">Calibration auto (px/mm)</div>
                </div>
                """, unsafe_allow_html=True)
                pixels_per_mm_adv = auto_cal
        
        # Launch analysis
        if st.button("🚀 Lancer l'Analyse Complète + Friction"):
            
            with st.spinner("🧮 Calcul des métriques avancées et analyse de friction..."):
                # Calculate metrics
                metrics = calculate_advanced_metrics(df_valid, fps_adv, pixels_per_mm_adv, mass_g, angle_deg_adv)
                friction_results = calculate_friction_coefficients(
                    df_valid, mass_g, angle_deg_adv, fps_adv, pixels_per_mm_adv
                )
            
            if metrics and friction_results:
                # === AFFICHAGE DES INFORMATIONS DE NETTOYAGE ===
                st.markdown("### 🧹 Nettoyage des Données")
                
                cleaning_info = metrics.get('cleaning_info', {})
                clean_col1, clean_col2, clean_col3 = st.columns(3)
                
                with clean_col1:
                    st.markdown(f"""
                    <div class="metric-item">
                        <div class="metric-value">{cleaning_info.get('original_length', 0)}</div>
                        <div class="metric-label">Points Originaux</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                with clean_col2:
                    st.markdown(f"""
                    <div class="metric-item">
                        <div class="metric-value">{cleaning_info.get('cleaned_length', 0)}</div>
                        <div class="metric-label">Points Nettoyés</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                with clean_col3:
                    percentage_kept = cleaning_info.get('percentage_kept', 0)
                    st.markdown(f"""
                    <div class="metric-item">
                        <div class="metric-value">{percentage_kept:.1f}%</div>
                        <div class="metric-label">Données Conservées</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                if cleaning_info.get('start_removed', 0) > 0 or cleaning_info.get('end_removed', 0) > 0:
                    st.success(f"✅ Artefacts supprimés : {cleaning_info.get('start_removed', 0)} points au début, {cleaning_info.get('end_removed', 0)} points à la fin")
                
                # === FRICTION ANALYSIS SECTION ===
                st.markdown("### 🔥 Analyse de Friction Grain-Sphère")
                
                # Display friction results
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
                
                # Advanced visualizations
                st.markdown("### 📈 Visualisations Avancées + Analyse de Friction")
                
                fig_advanced = make_subplots(
                    rows=2, cols=2,
                    subplot_titles=('Vitesse vs Temps', 'Accélération vs Temps',
                                   'Coefficients de Friction μ', 'Forces'),
                    vertical_spacing=0.1
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
                
                # 3. Friction coefficients
                fig_advanced.add_trace(
                    go.Scatter(x=friction_results['time'], y=friction_results['mu_kinetic_series'], 
                              mode='lines', name='μ cinétique',
                              line=dict(color='darkred', width=2)),
                    row=2, col=1
                )
                fig_advanced.add_trace(
                    go.Scatter(x=friction_results['time'], y=friction_results['mu_rolling_series'], 
                              mode='lines', name='μ roulement',
                              line=dict(color='orange', width=2)),
                    row=2, col=1
                )
                
                # 4. Forces plot
                fig_advanced.add_trace(
                    go.Scatter(x=metrics['time'], y=metrics['resistance_force']*1000, mode='lines', 
                             line=dict(color='red', width=2), name='F_résistance'),
                    row=2, col=2
                )
                
                # Update layout
                fig_advanced.update_layout(height=600, showlegend=False)
                fig_advanced.update_xaxes(title_text="Temps (s)")
                fig_advanced.update_yaxes(title_text="Vitesse (mm/s)", row=1, col=1)
                fig_advanced.update_yaxes(title_text="Accélération (mm/s²)", row=1, col=2)
                fig_advanced.update_yaxes(title_text="Coefficient de Friction", row=2, col=1)
                fig_advanced.update_yaxes(title_text="Force (mN)", row=2, col=2)
                
                st.plotly_chart(fig_advanced, use_container_width=True)
                
                # Results summary
                st.markdown("### 📊 Résumé des Résultats")
                
                summary_col1, summary_col2, summary_col3 = st.columns(3)
                
                with summary_col1:
                    st.markdown("**Cinématique**")
                    st.write(f"• Krr: {metrics['krr']:.6f}")
                    st.write(f"• Vitesse max: {metrics['max_velocity']*1000:.1f} mm/s")
                    st.write(f"• Distance: {metrics['distance']*1000:.1f} mm")
                    
                with summary_col2:
                    st.markdown("**Friction**")
                    st.write(f"• μ cinétique: {friction_results['mu_kinetic_avg']:.4f}")
                    st.write(f"• μ roulement: {friction_results['mu_rolling_avg']:.4f}")
                    st.write(f"• μ énergétique: {mu_energetic_val:.4f}")
                    
                with summary_col3:
                    st.markdown("**Énergies**")
                    st.write(f"• Dissipée: {metrics['energy_dissipated']*1000:.2f} mJ")
                    st.write(f"• Efficacité: {metrics['energy_efficiency']:.1f}%")
                    
                    # Validation
                    if 0.03 <= metrics['krr'] <= 0.10:
                        st.markdown('<div class="status-success">✅ Krr cohérent avec littérature</div>', unsafe_allow_html=True)
                    else:
                        st.markdown('<div class="status-warning">⚠️ Krr à vérifier</div>', unsafe_allow_html=True)
                
                st.success("✅ Analyse de friction terminée!")
                
            else:
                st.error("❌ Impossible de calculer les métriques - données insuffisantes")
    
    # Placeholder for other analysis types
    elif analysis_type == "📈 Code 1 : Visualisation de Trajectoire":
        st.markdown("## 📈 Code 1 : Visualisation de Trajectoire")
        st.info("Section en développement...")
        
    elif analysis_type == "📊 Code 2 : Analyse Krr":
        st.markdown("## 📊 Code 2 : Analyse Krr") 
        st.info("Section en développement...")
        
    elif analysis_type == "🔍 Comparaison Multi-Expériences":
        st.markdown("## 🔍 Comparaison Multi-Expériences")
        st.info("Section en développement...")

else:
    # No data loaded message
    st.markdown("""
    ## 🚀 Pour commencer
    
    Téléchargez vos données expérimentales ou utilisez les données d'exemple pour explorer la plateforme.
    
    ### 📋 Format de fichier attendu:
    - **Frame**: Numéro d'image
    - **X_center**: Position X du centre de la sphère
    - **Y_center**: Position Y du centre de la sphère  
    - **Radius**: Rayon détecté de la sphère
    
    ### 🔥 NOUVEAUTÉ - Analyse de Friction:
    - **μ cinétique**: Coefficient de friction cinétique grain-sphère
    - **μ roulement**: Coefficient de résistance au roulement pur
    - **μ énergétique**: Basé sur la dissipation d'énergie
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
### 🎓 Plateforme d'Analyse de Résistance au Roulement des Sphères
*Développée pour l'analyse de la résistance au roulement des sphères sur matériau granulaire humide*

**Institution:** Department of Cosmic Earth Science, Graduate School of Science, Osaka University  
**Innovation:** Première étude de l'effet de l'humidité + **🔥 Analyse de friction grain-sphère**
""")
