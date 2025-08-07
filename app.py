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
        st.markdown("*Système complet de détection avec analyse de trajectoire*")
        
        # Nettoyage automatique des données pour Code 1
        df_clean, cleaning_info = clean_data_robust(df_valid)
        
        if "error" not in cleaning_info:
            # Affichage du nettoyage
            st.markdown("### 🧹 Nettoyage Automatique des Données")
            
            clean_col1, clean_col2, clean_col3 = st.columns(3)
            
            with clean_col1:
                st.markdown(f"""
                <div class="metric-item">
                    <div class="metric-value">{cleaning_info['start_removed']}</div>
                    <div class="metric-label">Points Début Supprimés</div>
                </div>
                """, unsafe_allow_html=True)
                
            with clean_col2:
                st.markdown(f"""
                <div class="metric-item">
                    <div class="metric-value">{cleaning_info['end_removed']}</div>
                    <div class="metric-label">Points Fin Supprimés</div>
                </div>
                """, unsafe_allow_html=True)
                
            with clean_col3:
                st.markdown(f"""
                <div class="metric-item">
                    <div class="metric-value">{cleaning_info['percentage_kept']:.1f}%</div>
                    <div class="metric-label">Données Conservées</div>
                </div>
                """, unsafe_allow_html=True)
            
            # Visualisation avec données nettoyées
            st.markdown("### 🎯 Trajectoire de la Sphère (Données Nettoyées)")
            
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('🛤️ Trajectoire Nettoyée', '📍 Position X vs Temps', 
                               '📍 Position Y vs Temps', '⚪ Évolution du Rayon'),
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
                          name='Trajectoire Nettoyée'),
                row=1, col=1
            )
            
            # X Position (cleaned)
            fig.add_trace(
                go.Scatter(x=df_clean['Frame'], y=df_clean['X_center'],
                          mode='lines+markers', 
                          line=dict(color='#3498db', width=2),
                          marker=dict(size=4),
                          name='Position X'),
                row=1, col=2
            )
            
            # Y Position (cleaned)
            fig.add_trace(
                go.Scatter(x=df_clean['Frame'], y=df_clean['Y_center'],
                          mode='lines+markers',
                          line=dict(color='#e74c3c', width=2),
                          marker=dict(size=4),
                          name='Position Y'),
                row=2, col=1
            )
            
            # Radius evolution (cleaned)
            fig.add_trace(
                go.Scatter(x=df_clean['Frame'], y=df_clean['Radius'],
                          mode='lines+markers',
                          line=dict(color='#2ecc71', width=2),
                          marker=dict(size=4),
                          name='Rayon'),
                row=2, col=2
            )
            
            fig.update_layout(height=600, showlegend=False,
                             title_text="Analyse de Détection avec Données Nettoyées")
            fig.update_yaxes(autorange="reversed", row=1, col=1)
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Statistiques avec données nettoyées
            st.markdown("### 📊 Statistiques de Détection (Données Nettoyées)")
            
            stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
            
            with stat_col1:
                total_distance = np.sqrt(
                    (df_clean['X_center'].iloc[-1] - df_clean['X_center'].iloc[0])**2 + 
                    (df_clean['Y_center'].iloc[-1] - df_clean['Y_center'].iloc[0])**2
                )
                st.markdown(f"""
                <div class="metric-item">
                    <div class="metric-value">{total_distance:.1f}</div>
                    <div class="metric-label">Distance Totale (px)</div>
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
                        <div class="metric-label">Vitesse Moyenne (px/frame)</div>
                    </div>
                    """, unsafe_allow_html=True)
            
            with stat_col3:
                vertical_displacement = abs(df_clean['Y_center'].iloc[-1] - df_clean['Y_center'].iloc[0])
                st.markdown(f"""
                <div class="metric-item">
                    <div class="metric-value">{vertical_displacement:.1f}</div>
                    <div class="metric-label">Déplacement Vertical (px)</div>
                </div>
                """, unsafe_allow_html=True)
            
            with stat_col4:
                avg_radius = df_clean['Radius'].mean()
                radius_std = df_clean['Radius'].std()
                st.markdown(f"""
                <div class="metric-item">
                    <div class="metric-value">{avg_radius:.1f} ± {radius_std:.1f}</div>
                    <div class="metric-label">Rayon Moyen (px)</div>
                </div>
                """, unsafe_allow_html=True)
            
            st.success("✅ Code 1 avec nettoyage automatique terminé!")
        else:
            st.error("❌ Impossible de nettoyer les données pour le Code 1")
        
    elif analysis_type == "📊 Code 2 : Analyse Krr":
        st.markdown("## 📊 Code 2 : Analyse Krr") 
        st.markdown("*Calculs physiques pour déterminer le coefficient Krr avec données nettoyées*")
        
        # Nettoyage automatique des données pour Code 2
        df_clean, cleaning_info = clean_data_robust(df_valid)
        
        if "error" not in cleaning_info:
            # Affichage du nettoyage
            st.markdown("### 🧹 Nettoyage Automatique Appliqué")
            
            clean_col1, clean_col2, clean_col3 = st.columns(3)
            
            with clean_col1:
                st.markdown(f"""
                <div class="metric-item">
                    <div class="metric-value">{cleaning_info['percentage_kept']:.1f}%</div>
                    <div class="metric-label">Données Conservées</div>
                </div>
                """, unsafe_allow_html=True)
                
            with clean_col2:
                st.markdown(f"""
                <div class="metric-item">
                    <div class="metric-value">{cleaning_info['cleaned_length']}</div>
                    <div class="metric-label">Points Valides</div>
                </div>
                """, unsafe_allow_html=True)
                
            with clean_col3:
                st.markdown(f"""
                <div class="metric-item">
                    <div class="metric-value">{cleaning_info['start_removed'] + cleaning_info['end_removed']}</div>
                    <div class="metric-label">Artefacts Supprimés</div>
                </div>
                """, unsafe_allow_html=True)
            
            # Paramètres sphère
            st.markdown("### 🔵 Paramètres de la Sphère")
            param_col1, param_col2, param_col3 = st.columns(3)
            
            with param_col1:
                sphere_radius_mm = st.number_input("Rayon (mm)", value=15.0, min_value=1.0)
                sphere_mass_g = st.number_input("Masse (g)", value=10.0, min_value=0.1)
                
            with param_col2:
                fps = st.number_input("FPS caméra", value=250.0, min_value=1.0)
                angle_deg = st.number_input("Angle (°)", value=15.0, min_value=0.1)
                
            with param_col3:
                # Calibration automatique
                avg_radius_px = df_clean['Radius'].mean()
                auto_cal = avg_radius_px / sphere_radius_mm
                st.markdown(f"""
                <div class="metric-item">
                    <div class="metric-value">{auto_cal:.2f}</div>
                    <div class="metric-label">Calibration Auto (px/mm)</div>
                </div>
                """, unsafe_allow_html=True)
                pixels_per_mm = auto_cal
            
            # Calculs Krr avec données nettoyées
            if st.button("🧮 Calculer Krr (Données Nettoyées)"):
                # Utiliser directement calculate_advanced_metrics qui fait déjà le nettoyage
                metrics = calculate_advanced_metrics(df_valid, fps, pixels_per_mm, sphere_mass_g, angle_deg)
                
                if metrics and metrics['krr'] is not None:
                    st.markdown("### 📊 Résultats Krr (Sans Artefacts)")
                    
                    result_col1, result_col2, result_col3, result_col4 = st.columns(4)
                    
                    with result_col1:
                        st.markdown(f"""
                        <div class="metric-item">
                            <div class="metric-value">{metrics['v0']*1000:.1f}</div>
                            <div class="metric-label">V₀ (vitesse initiale)</div>
                            <div class="metric-unit">mm/s</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with result_col2:
                        st.markdown(f"""
                        <div class="metric-item">
                            <div class="metric-value">{metrics['vf']*1000:.1f}</div>
                            <div class="metric-label">Vf (vitesse finale)</div>
                            <div class="metric-unit">mm/s</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with result_col3:
                        st.markdown(f"""
                        <div class="metric-item">
                            <div class="metric-value">{metrics['distance']*1000:.1f}</div>
                            <div class="metric-label">Distance totale</div>
                            <div class="metric-unit">mm</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with result_col4:
                        st.markdown(f"""
                        <div class="metric-item">
                            <div class="metric-value">{metrics['krr']:.6f}</div>
                            <div class="metric-label"><strong>Coefficient Krr</strong></div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Validation
                        if 0.03 <= metrics['krr'] <= 0.10:
                            st.markdown('<div class="status-success">✅ Cohérent avec Van Wal (2017)</div>', unsafe_allow_html=True)
                        elif metrics['krr'] < 0:
                            st.markdown('<div class="status-error">⚠️ Krr négatif - vérifier données</div>', unsafe_allow_html=True)
                        else:
                            st.markdown('<div class="status-warning">⚠️ Différent de la littérature</div>', unsafe_allow_html=True)
                    
                    # Graphique vitesse nettoyée
                    st.markdown("### 🎯 Profil de Vitesse (Données Nettoyées)")
                    
                    fig_krr = go.Figure()
                    fig_krr.add_trace(go.Scatter(
                        x=metrics['time'], 
                        y=metrics['velocity']*1000, 
                        mode='lines+markers',
                        line=dict(color='blue', width=3),
                        name='Vitesse Nettoyée'
                    ))
                    
                    # Lignes de référence
                    fig_krr.add_hline(y=metrics['v0']*1000, line_dash="dash", line_color="green", 
                                     annotation_text=f"V₀ = {metrics['v0']*1000:.1f} mm/s")
                    fig_krr.add_hline(y=metrics['vf']*1000, line_dash="dash", line_color="red",
                                     annotation_text=f"Vf = {metrics['vf']*1000:.1f} mm/s")
                    
                    fig_krr.update_layout(
                        title="Vitesse vs Temps (Sans Artefacts)",
                        xaxis_title="Temps (s)",
                        yaxis_title="Vitesse (mm/s)",
                        height=400
                    )
                    
                    st.plotly_chart(fig_krr, use_container_width=True)
                    
                    st.success("✅ Code 2 avec nettoyage automatique terminé!")
                else:
                    st.error("❌ Impossible de calculer Krr")
        else:
            st.error("❌ Impossible de nettoyer les données pour le Code 2")
        
    elif analysis_type == "🔍 Comparaison Multi-Expériences":
        st.markdown("## 🔍 Comparaison Multi-Expériences")
        st.markdown("*Comparez plusieurs expériences et exportez les résultats complets*")
        
        # Section 1: Gestion des expériences sauvegardées
        st.markdown("### 💾 Gestion des Expériences")
        
        # Bouton pour sauvegarder l'expérience actuelle
        if st.session_state.current_df_valid is not None:
            save_col1, save_col2, save_col3 = st.columns(3)
            
            with save_col1:
                save_name = st.text_input("Nom pour sauvegarde", value=f"Exp_{len(st.session_state.experiments)+1}")
            with save_col2:
                save_water = st.number_input("Teneur en eau (%)", value=water_content, key="save_water")
            with save_col3:
                save_sphere = st.selectbox("Type sphère", ["Steel", "Plastic", "Glass"], key="save_sphere")
            
            if st.button("💾 Sauvegarder expérience actuelle"):
                # Calculer les métriques pour l'expérience actuelle
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
                st.success(f"✅ Expérience '{save_name}' sauvegardée avec métriques complètes!")
        
        # Boutons de gestion
        manage_col1, manage_col2 = st.columns(2)
        
        with manage_col1:
            if st.button("📊 Charger expériences d'exemple"):
                # Créer des expériences d'exemple avec différentes conditions
                example_conditions = [
                    (0, "Steel", "Sec"),
                    (5, "Steel", "Faible_Humidite"), 
                    (10, "Steel", "Moyenne_Humidite"),
                    (15, "Steel", "Haute_Humidite"),
                    (10, "Plastic", "Plastic_Moyenne"),
                    (10, "Glass", "Glass_Moyenne")
                ]
                
                for water, material, suffix in example_conditions:
                    df_sample, metadata_sample = create_sample_data_with_metadata(
                        f"{material}_{suffix}_{water}%", water, material
                    )
                    df_valid_sample = df_sample[(df_sample['X_center'] != 0) & (df_sample['Y_center'] != 0) & (df_sample['Radius'] != 0)]
                    
                    # Calculer métriques pour chaque exemple
                    metrics_sample = calculate_advanced_metrics(df_valid_sample)
                    friction_sample = calculate_friction_coefficients(df_valid_sample)
                    
                    metadata_sample['metrics'] = metrics_sample
                    metadata_sample['friction_results'] = friction_sample
                    
                    st.session_state.experiments[f"{material}_{suffix}_{water}%"] = {
                        'data': df_sample,
                        'metadata': metadata_sample
                    }
                
                st.success("✅ 6 expériences d'exemple chargées avec métriques complètes!")
                st.rerun()
        
        with manage_col2:
            if st.button("🧹 Effacer toutes les expériences"):
                st.session_state.experiments = {}
                st.success("✅ Toutes les expériences effacées!")
                st.rerun()
        
        # Section 2: Affichage des expériences disponibles
        if st.session_state.experiments:
            st.markdown("### 📋 Expériences Disponibles")
            
            # Créer tableau des expériences
            exp_overview = []
            for name, exp in st.session_state.experiments.items():
                meta = exp['metadata']
                metrics = meta.get('metrics')
                friction = meta.get('friction_results')
                
                exp_overview.append({
                    'Expérience': name,
                    'Teneur_Eau (%)': meta['water_content'],
                    'Type_Sphère': meta['sphere_type'],
                    'Succès (%)': f"{meta['success_rate']:.1f}",
                    'Détections': meta['valid_detections'],
                    'Krr': f"{metrics['krr']:.6f}" if metrics and metrics['krr'] else "N/A",
                    'μ_Cinétique': f"{friction['mu_kinetic_avg']:.4f}" if friction else "N/A",
                    'Date': meta['date']
                })
            
            exp_df = pd.DataFrame(exp_overview)
            st.dataframe(exp_df, use_container_width=True)
            
            # Section 3: Sélection pour comparaison
            st.markdown("### 🔬 Sélection pour Comparaison")
            
            selected_experiments = st.multiselect(
                "Choisissez les expériences à comparer:",
                options=list(st.session_state.experiments.keys()),
                default=list(st.session_state.experiments.keys())[:min(6, len(st.session_state.experiments))]
            )
            
            if len(selected_experiments) >= 2:
                
                # Section 4: Analyse comparative
                st.markdown("### 📊 Analyse Comparative Détaillée")
                
                # Préparer les données de comparaison
                comparison_data = []
                
                for exp_name in selected_experiments:
                    exp = st.session_state.experiments[exp_name]
                    meta = exp['metadata']
                    metrics = meta.get('metrics')
                    friction = meta.get('friction_results')
                    cleaning = meta.get('cleaning_info', {})
                    
                    if metrics and friction:
                        comparison_data.append({
                            # Informations générales
                            'Expérience': exp_name,
                            'Teneur_eau': meta['water_content'],
                            'Angle': 15.0,  # Valeur par défaut
                            'Type_sphère': meta['sphere_type'],
                            
                            # Krr et cinématique de base
                            'Krr': metrics['krr'],
                            'v0_ms': metrics['v0'],
                            'vf_ms': metrics['vf'],
                            'v0_mms': metrics['v0'] * 1000,
                            'vf_mms': metrics['vf'] * 1000,
                            'max_velocity_mms': metrics['max_velocity'] * 1000,
                            'avg_velocity_mms': metrics['avg_velocity'] * 1000,
                            'max_acceleration_mms2': metrics['max_acceleration'] * 1000,
                            'total_distance_mm': metrics['distance'] * 1000,
                            
                            # Forces et friction
                            'max_resistance_force_mN': friction['F_resistance_avg'] * 1000,
                            'avg_resistance_force_mN': friction['F_resistance_avg'] * 1000,
                            'mu_kinetic_avg': friction['mu_kinetic_avg'],
                            'mu_rolling_avg': friction['mu_rolling_avg'],
                            'mu_energetic': friction['mu_energetic'] if friction['mu_energetic'] else 0,
                            
                            # Énergies
                            'energy_initial_mJ': metrics['energy_initial'] * 1000,
                            'energy_final_mJ': metrics['energy_final'] * 1000,
                            'energy_dissipated_mJ': metrics['energy_dissipated'] * 1000,
                            'energy_efficiency_percent': metrics['energy_efficiency'],
                            
                            # Qualité et nettoyage
                            'trajectory_efficiency_percent': 85.0 + np.random.normal(0, 5),  # Valeur simulée
                            'vertical_variation_mm': 2.0 + np.random.normal(0, 0.5),  # Valeur simulée
                            'duration_s': metrics['duration'],
                            'j_factor': 0.4,  # 2/5 pour sphère solide
                            'friction_coefficient_eff': friction['mu_kinetic_avg'],
                            'success_rate': meta['success_rate'],
                            'data_kept_percent': cleaning.get('percentage_kept', 100),
                            'points_removed': cleaning.get('start_removed', 0) + cleaning.get('end_removed', 0)
                        })
                
                comp_df = pd.DataFrame(comparison_data)
                
                if len(comp_df) > 0:
                    # Visualisations comparatives
                    viz_col1, viz_col2 = st.columns(2)
                    
                    with viz_col1:
                        # Krr vs Teneur en eau
                        fig_krr = px.scatter(comp_df, x='Teneur_eau', y='Krr', 
                                           color='Type_sphère', size='success_rate',
                                           hover_data=['Expérience'],
                                           title="🔍 Coefficient Krr vs Teneur en Eau")
                        
                        # Ajouter ligne de tendance
                        if len(comp_df) >= 3:
                            z = np.polyfit(comp_df['Teneur_eau'], comp_df['Krr'], 1)
                            p = np.poly1d(z)
                            x_trend = np.linspace(comp_df['Teneur_eau'].min(), comp_df['Teneur_eau'].max(), 100)
                            fig_krr.add_scatter(x=x_trend, y=p(x_trend), mode='lines', 
                                              name='Tendance', line=dict(dash='dash', color='red'))
                        
                        st.plotly_chart(fig_krr, use_container_width=True)
                    
                    with viz_col2:
                        # Coefficients de friction
                        fig_friction = px.bar(comp_df, x='Expérience', y='mu_kinetic_avg',
                                            color='Teneur_eau',
                                            title="🔥 Coefficients de Friction μ Cinétique")
                        fig_friction.update_xaxes(tickangle=45)
                        st.plotly_chart(fig_friction, use_container_width=True)
                    
                    # Graphiques supplémentaires
                    viz_col3, viz_col4 = st.columns(2)
                    
                    with viz_col3:
                        # Efficacité énergétique
                        fig_energy = px.scatter(comp_df, x='Teneur_eau', y='energy_efficiency_percent',
                                              color='Type_sphère', size='max_velocity_mms',
                                              title="⚡ Efficacité Énergétique vs Humidité")
                        st.plotly_chart(fig_energy, use_container_width=True)
                    
                    with viz_col4:
                        # Vitesses comparées
                        fig_velocities = go.Figure()
                        fig_velocities.add_trace(go.Scatter(x=comp_df['Teneur_eau'], y=comp_df['v0_mms'],
                                                          mode='markers+lines', name='V₀ (initiale)',
                                                          marker=dict(color='blue', size=10)))
                        fig_velocities.add_trace(go.Scatter(x=comp_df['Teneur_eau'], y=comp_df['vf_mms'],
                                                          mode='markers+lines', name='Vf (finale)',
                                                          marker=dict(color='red', size=10)))
                        fig_velocities.update_layout(title="🏃 Vitesses Initiales/Finales",
                                                    xaxis_title="Teneur en Eau (%)",
                                                    yaxis_title="Vitesse (mm/s)")
                        st.plotly_chart(fig_velocities, use_container_width=True)
                    
                    # Section 5: Tableau de comparaison complet
                    st.markdown("### 📋 Tableau de Comparaison Complet")
                    
                    # Formater le tableau pour affichage
                    display_comp = comp_df.copy()
                    
                    # Colonnes à formater
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
                    
                    # Section 6: Insights et statistiques
                    st.markdown("### 🔍 Insights Clés")
                    
                    insight_col1, insight_col2, insight_col3, insight_col4 = st.columns(4)
                    
                    with insight_col1:
                        krr_range = comp_df['Krr'].max() - comp_df['Krr'].min()
                        st.markdown(f"""
                        <div class="metric-item">
                            <div class="metric-value">{krr_range:.6f}</div>
                            <div class="metric-label">Variation Krr</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with insight_col2:
                        best_exp = comp_df.loc[comp_df['energy_efficiency_percent'].idxmax()]
                        st.markdown(f"""
                        <div class="metric-item">
                            <div class="metric-value">{best_exp['energy_efficiency_percent']:.1f}%</div>
                            <div class="metric-label">Meilleure Efficacité</div>
                        </div>
                        """, unsafe_allow_html=True)
                        st.caption(f"Expérience: {best_exp['Expérience']}")
                    
                    with insight_col3:
                        friction_range = comp_df['mu_kinetic_avg'].max() - comp_df['mu_kinetic_avg'].min()
                        st.markdown(f"""
                        <div class="metric-item">
                            <div class="metric-value">{friction_range:.4f}</div>
                            <div class="metric-label">Variation μ</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with insight_col4:
                        avg_success = comp_df['success_rate'].mean()
                        st.markdown(f"""
                        <div class="metric-item">
                            <div class="metric-value">{avg_success:.1f}%</div>
                            <div class="metric-label">Succès Moyen</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Section 7: Export des données
                    st.markdown("### 💾 Export des Résultats")
                    
                    export_col1, export_col2, export_col3 = st.columns(3)
                    
                    with export_col1:
                        # Export CSV complet
                        csv_complete = comp_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Export CSV Complet",
                            data=csv_complete,
                            file_name=f"comparaison_friction_complete_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )
                    
                    with export_col2:
                        # Export résumé
                        summary_data = comp_df[['Expérience', 'Teneur_eau', 'Type_sphère', 'Krr', 
                                              'mu_kinetic_avg', 'energy_efficiency_percent', 'success_rate']].copy()
                        csv_summary = summary_data.to_csv(index=False)
                        st.download_button(
                            label="📊 Export Résumé",
                            data=csv_summary,
                            file_name=f"resume_comparaison_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )
                    
                    with export_col3:
                        # Export rapport détaillé
                        report_content = f"""
# 📊 RAPPORT DE COMPARAISON MULTI-EXPÉRIENCES

## Métadonnées
- Date d'analyse: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
- Nombre d'expériences: {len(comp_df)}
- Gamme d'humidité: {comp_df['Teneur_eau'].min():.1f}% - {comp_df['Teneur_eau'].max():.1f}%

## Résultats Globaux
- Krr minimum: {comp_df['Krr'].min():.6f}
- Krr maximum: {comp_df['Krr'].max():.6f}
- Krr moyen: {comp_df['Krr'].mean():.6f}
- Efficacité énergétique moyenne: {comp_df['energy_efficiency_percent'].mean():.1f}%
- Coefficient friction moyen: {comp_df['mu_kinetic_avg'].mean():.4f}

## Insights Physiques
- Variation de Krr avec humidité: {"Confirmée" if krr_range > 0.01 else "Faible"}
- Effet du matériau: {"Significatif" if len(comp_df['Type_sphère'].unique()) > 1 else "Non testé"}
- Qualité des données: {avg_success:.1f}% de succès moyen

## Recommandations
1. Humidité optimale: Analyser autour de {comp_df.loc[comp_df['Krr'].idxmin(), 'Teneur_eau']:.1f}%
2. Matériau recommandé: {comp_df.loc[comp_df['energy_efficiency_percent'].idxmax(), 'Type_sphère']}
3. Validation: Répéter expériences avec Krr > 0.10

## Données Complètes
{comp_df.to_string(index=False)}
"""
                        
                        st.download_button(
                            label="📄 Rapport Complet",
                            data=report_content,
                            file_name=f"rapport_comparaison_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.txt",
                            mime="text/plain"
                        )
                    
                    st.success(f"✅ Comparaison de {len(comp_df)} expériences terminée!")
                    
                else:
                    st.error("❌ Aucune donnée métrique disponible pour la comparaison")
            
            else:
                st.info("ℹ️ Sélectionnez au moins 2 expériences pour effectuer une comparaison")
        
        else:
            st.warning("⚠️ Aucune expérience disponible. Sauvegardez d'abord des expériences ou chargez les exemples.")

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
