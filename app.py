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

# Initialize session state for multi-experiment comparison
if 'experiments' not in st.session_state:
    st.session_state.experiments = {}

# Custom CSS
st.markdown("""
<style>
    .upload-section {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 1rem 0;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .section-header {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 5px;
        border-left: 4px solid #ff6b6b;
        margin: 1rem 0;
    }
    .comparison-card {
        background: linear-gradient(135deg, #00c6ff 0%, #0072ff 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
    .prediction-card {
        background: linear-gradient(135deg, #ff9a56 0%, #ff6b9d 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
        text-align: center;
    }
    .warning-card {
        background: linear-gradient(135deg, #ffeaa7 0%, #fab1a0 100%);
        padding: 1rem;
        border-radius: 10px;
        color: #2d3436;
        margin: 0.5rem 0;
    }
    .analysis-card {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid #00b894;
    }
</style>
""", unsafe_allow_html=True)

# ==================== HELPER FUNCTIONS ====================

def calculate_r2(y_true, y_pred):
    """Calculate R-squared score"""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

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
    """Calculate advanced kinematic and dynamic metrics for comparison"""
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
    
    # Detection quality
    radius_variation = df_valid['Radius'].std()
    detection_gaps = np.sum(np.diff(df_valid['Frame']) > 1)
    
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
        
        # Detection quality metrics
        'radius_variation': radius_variation,
        'detection_gaps': detection_gaps,
        
        # Time series for plotting
        'time': t,
        'velocity': v_magnitude,
        'acceleration': acceleration,
        'resistance_force': F_resistance,
        'power': P_resistance,
        'energy_kinetic': E_kinetic
    }

def build_prediction_model(experiments_data):
    """Build predictive models from experimental data"""
    if not experiments_data or len(experiments_data) < 3:
        return None
    
    # Collect all data points
    all_data = []
    for exp_name, exp in experiments_data.items():
        df = exp['data']
        meta = exp['metadata']
        df_valid = df[(df['X_center'] != 0) & (df['Y_center'] != 0) & (df['Radius'] != 0)]
        
        metrics = calculate_advanced_metrics(df_valid)
        if metrics and metrics['krr'] is not None:
            all_data.append({
                'water_content': meta['water_content'],
                'sphere_type_steel': 1 if meta['sphere_type'] == 'Steel' else 0,
                'sphere_type_plastic': 1 if meta['sphere_type'] == 'Plastic' else 0,
                'krr': metrics['krr'],
                'max_velocity': metrics['max_velocity'],
                'max_acceleration': metrics['max_acceleration'],
                'energy_efficiency': metrics['energy_efficiency'],
                'trajectory_efficiency': metrics['trajectory_efficiency'],
                'resistance_force': metrics['avg_resistance_force']
            })
    
    if len(all_data) < 3:
        return None
    
    df_model = pd.DataFrame(all_data)
    models = {}
    
    # Krr prediction model
    if df_model['krr'].notna().sum() >= 3:
        x_water = df_model['water_content'].values
        y_krr = df_model['krr'].values
        
        # Remove NaN values
        mask = ~(np.isnan(x_water) | np.isnan(y_krr))
        x_clean = x_water[mask]
        y_clean = y_krr[mask]
        
        if len(x_clean) >= 3:
            # Fit polynomial (degree 2 if enough points, else linear)
            degree = 2 if len(x_clean) >= 4 else 1
            coeffs = np.polyfit(x_clean, y_clean, degree)
            
            # Calculate R² and standard error
            y_pred = np.polyval(coeffs, x_clean)
            r2 = calculate_r2(y_clean, y_pred)
            std_error = np.std(y_clean - y_pred)
            
            models['krr'] = {
                'coeffs': coeffs,
                'degree': degree,
                'r2': r2,
                'data_range': (x_clean.min(), x_clean.max()),
                'std_error': std_error,
                'data_points': len(x_clean)
            }
    
    # Energy efficiency model
    if df_model['energy_efficiency'].notna().sum() >= 3:
        x_water = df_model['water_content'].values
        y_energy = df_model['energy_efficiency'].values
        
        mask = ~(np.isnan(x_water) | np.isnan(y_energy))
        x_clean = x_water[mask]
        y_clean = y_energy[mask]
        
        if len(x_clean) >= 3:
            degree = 2 if len(x_clean) >= 4 else 1
            coeffs = np.polyfit(x_clean, y_clean, degree)
            
            y_pred = np.polyval(coeffs, x_clean)
            r2 = calculate_r2(y_clean, y_pred)
            std_error = np.std(y_clean - y_pred)
            
            models['energy_efficiency'] = {
                'coeffs': coeffs,
                'degree': degree,
                'r2': r2,
                'data_range': (x_clean.min(), x_clean.max()),
                'std_error': std_error,
                'data_points': len(x_clean)
            }
    
    return models

def predict_with_confidence(model, water_content, confidence_level=0.95):
    """Predict value with confidence interval"""
    if not model:
        return None, None, None, True
    
    # Check if water content is within data range
    min_water, max_water = model['data_range']
    extrapolation = water_content < min_water or water_content > max_water
    
    # Make prediction
    prediction = np.polyval(model['coeffs'], water_content)
    
    # Calculate confidence interval (simplified)
    z_score = 1.96 if confidence_level == 0.95 else 1.645  # 95% or 90%
    margin_error = z_score * model['std_error']
    
    ci_lower = prediction - margin_error
    ci_upper = prediction + margin_error
    
    return prediction, ci_lower, ci_upper, extrapolation

def generate_engineering_recommendations(experiments_data, models):
    """Generate practical engineering recommendations"""
    recommendations = []
    
    if not experiments_data or len(experiments_data) < 2:
        return ["Données insuffisantes pour des recommandations fiables. Besoin d'au moins 2 expériences."]
    
    # Analyze water content effects
    water_contents = []
    krr_values = []
    
    for exp_name, exp in experiments_data.items():
        df = exp['data']
        meta = exp['metadata']
        df_valid = df[(df['X_center'] != 0) & (df['Y_center'] != 0) & (df['Radius'] != 0)]
        
        metrics = calculate_advanced_metrics(df_valid)
        if metrics and metrics['krr']:
            water_contents.append(meta['water_content'])
            krr_values.append(metrics['krr'])
    
    if len(water_contents) >= 2:
        # Find optimal water content
        if models and 'krr' in models:
            # Use model to find minimum
            water_range = np.linspace(min(water_contents), max(water_contents), 100)
            krr_predictions = [np.polyval(models['krr']['coeffs'], w) for w in water_range]
            optimal_idx = np.argmin(krr_predictions)
            optimal_water = water_range[optimal_idx]
            optimal_krr = krr_predictions[optimal_idx]
            
            recommendations.append(f"🎯 **Teneur en eau optimale**: {optimal_water:.1f}% (Krr prédit = {optimal_krr:.6f})")
        else:
            # Simple analysis
            min_krr_idx = np.argmin(krr_values)
            optimal_water = water_contents[min_krr_idx]
            optimal_krr = krr_values[min_krr_idx]
            recommendations.append(f"🎯 **Meilleures conditions observées**: {optimal_water:.1f}% d'eau (Krr = {optimal_krr:.6f})")
        
        # Practical thresholds
        max_krr = max(krr_values)
        min_krr = min(krr_values)
        krr_increase = (max_krr - min_krr) / min_krr * 100
        
        if krr_increase > 50:
            recommendations.append(f"⚠️ **Sensibilité critique**: {krr_increase:.0f}% d'augmentation de résistance")
        elif krr_increase > 20:
            recommendations.append(f"⚠️ **Sensibilité modérée**: {krr_increase:.0f}% d'augmentation de résistance")
        else:
            recommendations.append(f"✅ **Faible sensibilité**: Seulement {krr_increase:.0f}% d'augmentation de résistance")
    
    return recommendations

# ==================== MAIN APPLICATION ====================

# Page navigation
st.sidebar.markdown("### 📋 Navigation")
page = st.sidebar.radio("Sélectionner la Page:", [
    "🏠 Analyse Unique Détaillée",
    "🔍 Comparaison Multi-Expériences", 
    "🎯 Module de Prédiction",
    "📊 Rapport Auto-Généré"
])

# ==================== MAIN SINGLE ANALYSIS PAGE ====================
if page == "🏠 Analyse Unique Détaillée":
    st.markdown("""
    # ⚪ Plateforme d'Analyse de Résistance au Roulement des Sphères
    ## 🔬 Suite d'Analyse Complète pour la Recherche en Mécanique Granulaire
    *Analyses détaillées avec visualisation, calcul Krr et métriques avancées*
    """)

    # File upload section
    st.markdown("""
    <div class="upload-section">
        <h2>📂 Téléchargez Vos Données Expérimentales</h2>
        <p>Commencez par télécharger votre fichier CSV avec les résultats de détection</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Experiment metadata input for saving
    col1, col2, col3 = st.columns(3)
    with col1:
        experiment_name = st.text_input("Nom de l'Expérience", value="Experiment_1")
    with col2:
        water_content = st.number_input("Teneur en Eau (%)", value=0.0, min_value=0.0, max_value=30.0)
    with col3:
        sphere_type = st.selectbox("Type de Sphère", ["Steel", "Plastic", "Glass", "Other"])

    # Initialize data variables
    df = None
    df_valid = None
    
    # Create tabs for different input methods
    tab1, tab2 = st.tabs(["📁 Upload Fichier", "🔬 Données d'Exemple"])
    
    with tab1:
        st.markdown("### 📁 Télécharger un Fichier CSV")
        
        # File upload
        uploaded_file = st.file_uploader(
            "Choisissez votre fichier CSV avec les données de détection", 
            type=['csv'],
            help="Téléchargez un fichier CSV avec les colonnes: Frame, X_center, Y_center, Radius",
            key="file_uploader"
        )
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.success(f"✅ Fichier lu avec succès! {len(df)} lignes trouvées")
                
                # Check required columns
                required_columns = ['Frame', 'X_center', 'Y_center', 'Radius']
                missing_columns = [col for col in required_columns if col not in df.columns]
                
                if missing_columns:
                    st.error(f"❌ Colonnes manquantes: {missing_columns}")
                    df = None
                else:
                    df_valid = df[(df['X_center'] != 0) & (df['Y_center'] != 0) & (df['Radius'] != 0)]
                    st.session_state['current_df'] = df
                    st.session_state['current_df_valid'] = df_valid
                    
            except Exception as e:
                st.error(f"❌ Erreur lors du chargement du fichier: {str(e)}")
    
    with tab2:
        st.markdown("### 🔬 Utiliser des Données d'Exemple")
        
        col1, col2 = st.columns(2)
        with col1:
            sample_water = st.slider("Teneur en Eau d'Exemple (%)", 0.0, 25.0, 10.0, 0.5)
        with col2:
            sample_sphere = st.selectbox("Type de Sphère d'Exemple", ["Steel", "Plastic", "Glass"])
        
        if st.button("🔬 Générer des données d'exemple", key="generate_sample"):
            df, metadata = create_sample_data_with_metadata(
                experiment_name=f"Sample_{sample_water}%", 
                water_content=sample_water, 
                sphere_type=sample_sphere
            )
            df_valid = df[(df['X_center'] != 0) & (df['Y_center'] != 0) & (df['Radius'] != 0)]
            
            st.session_state['current_df'] = df
            st.session_state['current_df_valid'] = df_valid
            
            st.success("📊 Données d'exemple générées avec succès!")

    # Use data from session state if available
    if 'current_df' in st.session_state and 'current_df_valid' in st.session_state:
        df = st.session_state['current_df']
        df_valid = st.session_state['current_df_valid']
    
    # Analysis section - ALWAYS show if we have data
    if df is not None and df_valid is not None and len(df_valid) > 0:
        
        # Quick data overview
        st.markdown("### 📊 Aperçu de Vos Données")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h3>{len(df)}</h3>
                <p>Frames Totales</p>
            </div>
            """, unsafe_allow_html=True)
            
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <h3>{len(df_valid)}</h3>
                <p>Détections Valides</p>
            </div>
            """, unsafe_allow_html=True)
            
        with col3:
            success_rate = len(df_valid) / len(df) * 100 if len(df) > 0 else 0
            st.markdown(f"""
            <div class="metric-card">
                <h3>{success_rate:.1f}%</h3>
                <p>Taux de Succès</p>
            </div>
            """, unsafe_allow_html=True)
            
        with col4:
            avg_radius = df_valid['Radius'].mean() if len(df_valid) > 0 else 0
            st.markdown(f"""
            <div class="metric-card">
                <h3>{avg_radius:.1f} px</h3>
                <p>Rayon Moyen</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Add option to save experiment for comparison
        if st.button("💾 Sauvegarder l'expérience pour comparaison"):
            metadata = {
                'experiment_name': experiment_name,
                'water_content': water_content,
                'sphere_type': sphere_type,
                'date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
                'total_frames': len(df),
                'valid_detections': len(df_valid),
                'success_rate': len(df_valid) / len(df) * 100 if len(df) > 0 else 0
            }
            st.session_state.experiments[experiment_name] = {
                'data': df,
                'metadata': metadata
            }
            st.success(f"Expérience '{experiment_name}' sauvegardée pour comparaison!")
        
        # Navigation between the 3 detailed analysis codes
        st.markdown("---")
        st.markdown("## 🔧 Analyses Détaillées Disponibles")
        
        # Sidebar for navigation
        st.sidebar.title("🧭 Types d'Analyse")
        analysis_type = st.sidebar.selectbox("Sélectionnez le type d'analyse :", [
            "📈 Code 1 : Visualisation de Trajectoire",
            "📊 Code 2 : Analyse Krr Détaillée",
            "🔬 Code 3 : Analyse Complète Avancée",
            "📋 Vue d'ensemble des données"
        ])
        
        # === CODE 1: DETECTION AND TRAJECTORY VISUALIZATION ===
        if analysis_type == "📈 Code 1 : Visualisation de Trajectoire":
            st.markdown("""
            <div class="analysis-card">
                <h2>📈 Code 1 : Détection et Visualisation de Trajectoire</h2>
                <p>Système complet de détection de sphères avec analyse de trajectoire</p>
            </div>
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
            
            # Visualization of loaded data
            if len(df_valid) > 0:
                st.markdown("### 🎯 Trajectoire de la Sphère Détectée")
                
                # Main trajectory plot
                fig = make_subplots(
                    rows=2, cols=2,
                    subplot_titles=('🛤️ Trajectoire Complète', '📍 Position X vs Temps', 
                                   '📍 Position Y vs Temps', '⚪ Évolution du Rayon'),
                    specs=[[{"secondary_y": False}, {"secondary_y": False}],
                           [{"secondary_y": False}, {"secondary_y": False}]]
                )
                
                # Trajectory with color gradient based on time
                fig.add_trace(
                    go.Scatter(x=df_valid['X_center'], y=df_valid['Y_center'],
                              mode='markers+lines', 
                              marker=dict(color=df_valid['Frame'], 
                                        colorscale='viridis', 
                                        size=8,
                                        colorbar=dict(title="Frame")),
                              line=dict(width=2),
                              name='Trajectoire'),
                    row=1, col=1
                )
                
                # X Position
                fig.add_trace(
                    go.Scatter(x=df_valid['Frame'], y=df_valid['X_center'],
                              mode='lines+markers', 
                              line=dict(color='#3498db', width=3),
                              name='Position X'),
                    row=1, col=2
                )
                
                # Y Position
                fig.add_trace(
                    go.Scatter(x=df_valid['Frame'], y=df_valid['Y_center'],
                              mode='lines+markers',
                              line=dict(color='#e74c3c', width=3),
                              name='Position Y'),
                    row=2, col=1
                )
                
                # Detected radius
                fig.add_trace(
                    go.Scatter(x=df_valid['Frame'], y=df_valid['Radius'],
                              mode='lines+markers',
                              line=dict(color='#2ecc71', width=3),
                              name='Rayon'),
                    row=2, col=2
                )
                
                fig.update_layout(height=800, showlegend=False,
                                 title_text="Analyse Complète de Détection")
                
                # Reverse Y axis for trajectory (image coordinates)
                fig.update_yaxes(autorange="reversed", row=1, col=1)
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Detection statistics
                st.markdown("### 📊 Statistiques de Détection")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    total_distance = np.sqrt(
                        (df_valid['X_center'].iloc[-1] - df_valid['X_center'].iloc[0])**2 + 
                        (df_valid['Y_center'].iloc[-1] - df_valid['Y_center'].iloc[0])**2
                    )
                    st.metric("Distance Totale", f"{total_distance:.1f} px")
                    
                with col2:
                    if len(df_valid) > 1:
                        dx = df_valid['X_center'].diff()
                        dy = df_valid['Y_center'].diff()
                        speed = np.sqrt(dx**2 + dy**2)
                        avg_speed = speed.mean()
                        st.metric("Vitesse Moyenne", f"{avg_speed:.2f} px/frame")
                        
                with col3:
                    vertical_displacement = abs(df_valid['Y_center'].iloc[-1] - df_valid['Y_center'].iloc[0])
                    st.metric("Déplacement Vertical", f"{vertical_displacement:.1f} px")
                    
                with col4:
                    avg_radius = df_valid['Radius'].mean()
                    radius_std = df_valid['Radius'].std()
                    st.metric("Rayon Moyen", f"{avg_radius:.1f} ± {radius_std:.1f} px")
                
                # Detection quality analysis
                st.markdown("### 🔍 Qualité de Détection")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    fig_radius = px.histogram(df_valid, x='Radius', nbins=15,
                                             title="Distribution des Rayons Détectés",
                                             labels={'Radius': 'Rayon (pixels)', 'count': 'Fréquence'})
                    fig_radius.add_vline(x=minR, line_dash="dash", line_color="red", 
                                        annotation_text=f"Min: {minR}")
                    fig_radius.add_vline(x=maxR, line_dash="dash", line_color="red", 
                                        annotation_text=f"Max: {maxR}")
                    st.plotly_chart(fig_radius, use_container_width=True)
                    
                with col2:
                    # Movement continuity analysis
                    if len(df_valid) > 1:
                        dx = df_valid['X_center'].diff()
                        dy = df_valid['Y_center'].diff()
                        movement = np.sqrt(dx**2 + dy**2)
                        movement_clean = movement.dropna()
                        frames_clean = df_valid['Frame'][1:len(movement_clean)+1]
                        
                        fig_movement = go.Figure()
                        fig_movement.add_trace(go.Scatter(
                            x=frames_clean, 
                            y=movement_clean,
                            mode='lines+markers',
                            name='Mouvement',
                            line=dict(color='blue', width=2)
                        ))
                        fig_movement.add_hline(y=max_movement, line_dash="dash", line_color="red",
                                              annotation_text=f"Max autorisé: {max_movement}")
                        fig_movement.update_layout(
                            title="Mouvement Inter-Frame",
                            xaxis_title="Frame",
                            yaxis_title="Déplacement (pixels)",
                            height=400
                        )
                        st.plotly_chart(fig_movement, use_container_width=True)
            
            # Information about detection algorithm
            st.markdown("### 🧠 Algorithme de Détection")
            st.markdown(f"""
            **Méthode utilisée :** Détection de cercles par soustraction de fond
            
            **Étapes principales :**
            1. **Création du fond** : Moyenne de 150 images de référence
            2. **Soustraction** : Élimination du fond statique
            3. **Seuillage** : Binarisation avec seuil adaptatif
            4. **Morphologie** : Nettoyage des contours
            5. **Détection** : Recherche de contours circulaires
            6. **Validation** : Filtrage par taille, forme et continuité
            
            **Critères de qualité :**
            - Taille : {minR} ≤ rayon ≤ {maxR} pixels
            - Forme : Circularité ≥ {circularity_min}
            - Continuité : Mouvement ≤ {max_movement} pixels/frame
            - Score : Qualité globale ≥ {min_score}
            """)
        
        # === CODE 2: KRR ANALYSIS ===
        elif analysis_type == "📊 Code 2 : Analyse Krr Détaillée":
            st.markdown("""
            <div class="analysis-card">
                <h2>📊 Code 2 : Analyse du Coefficient de Résistance au Roulement (Krr)</h2>
                <p>Calculs physiques complets pour déterminer le coefficient Krr</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Sphere parameters
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
                st.metric("Densité", f"{density_kg_m3:.0f} kg/m³")
                
            with col3:
                st.metric("Facteur d'inertie j", f"{j_value:.3f}")
                st.metric("Facteur (1+j)⁻¹", f"{1/(1+j_value):.4f}")
            
            # Experimental parameters
            st.markdown("### 📐 Paramètres Expérimentaux")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                fps = st.number_input("FPS de la caméra", value=250.0, min_value=1.0, max_value=1000.0)
                angle_deg = st.number_input("Angle d'inclinaison (°)", value=15.0, min_value=0.1, max_value=45.0)
                
            with col2:
                # Automatic calibration based on detected radius
                if len(df_valid) > 0:
                    avg_radius_pixels = df_valid['Radius'].mean()
                    auto_calibration = avg_radius_pixels / sphere_radius_mm
                    st.metric("Calibration auto", f"{auto_calibration:.2f} px/mm")
                    
                    use_auto_cal = st.checkbox("Utiliser calibration automatique", value=True)
                    if use_auto_cal:
                        pixels_per_mm = auto_calibration
                    else:
                        pixels_per_mm = st.number_input("Calibration (px/mm)", value=auto_calibration, min_value=0.1)
                else:
                    pixels_per_mm = st.number_input("Calibration (px/mm)", value=5.0, min_value=0.1)
                    
            with col3:
                water_content_analysis = st.number_input("Teneur en eau (%)", value=water_content, min_value=0.0, max_value=100.0, key="krr_water")
                
            # Kinematic calculations and Krr
            if len(df_valid) > 10:
                st.markdown("### 🧮 Calculs Cinématiques")
                
                # Unit conversion
                dt = 1 / fps  # s
                
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
                
                # Initial and final velocities (average over a few points)
                n_avg = min(3, len(v_magnitude)//4)
                v0 = np.mean(v_magnitude[:n_avg])
                vf = np.mean(v_magnitude[-n_avg:])
                
                # Total distance
                distances = np.sqrt(np.diff(x_m)**2 + np.diff(y_m)**2)
                total_distance = np.sum(distances)
                
                # Calculate Krr coefficient
                g = 9.81  # m/s²
                if total_distance > 0:
                    krr = (v0**2 - vf**2) / (2 * g * total_distance)
                    
                    # Effective friction coefficient
                    angle_rad = np.radians(angle_deg)
                    mu_eff = krr + np.tan(angle_rad)
                    
                    # Display results
                    st.markdown("### 📈 Résultats Krr")
                    
                    result_col1, result_col2, result_col3, result_col4 = st.columns(4)
                    
                    with result_col1:
                        st.metric("V₀ (vitesse initiale)", f"{v0*1000:.1f} mm/s")
                        st.caption(f"{v0:.4f} m/s")
                        
                    with result_col2:
                        st.metric("Vf (vitesse finale)", f"{vf*1000:.1f} mm/s") 
                        st.caption(f"{vf:.4f} m/s")
                        
                    with result_col3:
                        st.metric("Distance totale", f"{total_distance*1000:.1f} mm")
                        st.caption(f"{total_distance:.4f} m")
                        
                    with result_col4:
                        st.metric("**Coefficient Krr**", f"{krr:.6f}")
                        if 0.03 <= krr <= 0.10:
                            st.success("✅ Cohérent avec Van Wal (2017)")
                        elif krr < 0:
                            st.error("⚠️ Krr négatif - sphère accélère")
                        else:
                            st.warning("⚠️ Différent de la littérature")
                    
                    # Quick trajectory visualization
                    st.markdown("### 🎯 Trajectoire et Profil de Vitesse")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        fig_traj = px.scatter(df_valid, x='X_center', y='Y_center', 
                                           color='Frame', size='Radius',
                                           title="Trajectoire de la Sphère")
                        fig_traj.update_yaxes(autorange="reversed")
                        st.plotly_chart(fig_traj, use_container_width=True)
                    
                    with col2:
                        fig_vel = go.Figure()
                        fig_vel.add_trace(go.Scatter(x=t, y=v_magnitude*1000, 
                                                   mode='lines+markers',
                                                   name='Vitesse',
                                                   line=dict(color='blue', width=2)))
                        fig_vel.update_layout(
                            title="Vitesse vs Temps",
                            xaxis_title="Temps (s)",
                            yaxis_title="Vitesse (mm/s)"
                        )
                        st.plotly_chart(fig_vel, use_container_width=True)
                
                else:
                    st.error("❌ Distance parcourue nulle - impossible de calculer Krr")
            else:
                st.warning("⚠️ Pas assez de données valides pour l'analyse Krr")
        
        # === CODE 3: ADVANCED AND COMPLETE ANALYSIS ===
        elif analysis_type == "🔬 Code 3 : Analyse Complète Avancée":
            st.markdown("""
            <div class="analysis-card">
                <h2>🔬 Code 3 : Analyse Cinématique Avancée et Complète</h2>
                <p>Analyse approfondie avec debug et métriques avancées</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Data verification
            st.markdown("### 🔍 Vérification des Données")
            if len(df_valid) > 0:
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Données valides", f"{len(df_valid)} frames")
                    st.metric("Taux de succès", f"{len(df_valid)/len(df)*100:.1f}%")
                    
                with col2:
                    radius_range = df_valid['Radius'].max() - df_valid['Radius'].min()
                    st.metric("Variation de rayon", f"{radius_range:.1f} px")
                    st.metric("Première détection", f"Frame {df_valid['Frame'].min()}")
                    
                with col3:
                    st.metric("Dernière détection", f"Frame {df_valid['Frame'].max()}")
                    duration_frames = df_valid['Frame'].max() - df_valid['Frame'].min()
                    st.metric("Durée de suivi", f"{duration_frames} frames")
                
                # Parameters for advanced analysis
                st.markdown("### ⚙️ Paramètres d'Analyse Avancée")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown("**Paramètres Sphère**")
                    mass_g = st.number_input("Masse (g)", value=10.0, min_value=0.1, key="adv_mass")
                    radius_mm = st.number_input("Rayon (mm)", value=15.0, min_value=1.0, key="adv_radius")
                    sphere_type_adv = st.selectbox("Type", ["Solide", "Creuse"], key="adv_type")
                    j_factor = 2/5 if sphere_type_adv == "Solide" else 2/3
                    
                with col2:
                    st.markdown("**Paramètres Expérimentaux**")
                    fps_adv = st.number_input("FPS", value=250.0, min_value=1.0, key="adv_fps")
                    angle_deg_adv = st.number_input("Angle (°)", value=15.0, min_value=0.1, key="adv_angle")
                    
                    # Automatic calibration
                    if len(df_valid) > 0:
                        avg_radius_px = df_valid['Radius'].mean()
                        auto_cal = avg_radius_px / radius_mm
                        st.metric("Calibration auto", f"{auto_cal:.2f} px/mm")
                        pixels_per_mm_adv = auto_cal
                
                with col3:
                    st.markdown("**Filtrage des Données**")
                    use_smoothing = st.checkbox("Lissage des données", value=True)
                    smooth_window = st.slider("Fenêtre de lissage", 3, 11, 5, step=2)
                    remove_outliers = st.checkbox("Supprimer les aberrants", value=True)
                
                # Advanced kinematic calculations
                if st.button("🚀 Lancer l'Analyse Complète"):
                    
                    st.markdown("### 🧮 Calculs Cinématiques Avancés")
                    
                    with st.spinner("🧮 Calcul des métriques avancées en cours..."):
                        metrics = calculate_advanced_metrics(df_valid, fps_adv, pixels_per_mm_adv, mass_g, angle_deg_adv)
                    
                    if metrics and metrics['krr'] is not None:
                        # Display main results
                        col1, col2, col3 = st.columns(3)
                        
                        with col2:  # Center the main result
                            st.markdown(f"""
                            <div class="prediction-card">
                                <h4>🎯 Résultat Principal</h4>
                                <h2>Krr = {metrics['krr']:.6f}</h2>
                                <p>Vitesse Max: {metrics['max_velocity']*1000:.1f} mm/s</p>
                                <p>Distance: {metrics['distance']*1000:.1f} mm</p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # Additional metrics in a grid
                        st.markdown("#### 📊 Métriques Détaillées")
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("Distance Totale", f"{metrics['distance']*1000:.1f} mm")
                            st.metric("Vitesse Initiale", f"{metrics['v0']*1000:.1f} mm/s")
                        with col2:
                            st.metric("Durée", f"{metrics['duration']:.2f} s")
                            st.metric("Vitesse Finale", f"{metrics['vf']*1000:.1f} mm/s")
                        with col3:
                            st.metric("Efficacité Énergétique", f"{metrics['energy_efficiency']:.1f}%")
                            st.metric("Accélération Max", f"{metrics['max_acceleration']*1000:.1f} mm/s²")
                        with col4:
                            st.metric("Efficacité Trajectoire", f"{metrics['trajectory_efficiency']:.1f}%")
                            st.metric("Force Résistance Max", f"{metrics['max_resistance_force']*1000:.1f} mN")
                        
                        # Advanced visualization
                        st.markdown("#### 📈 Visualisations Cinématiques")
                        
                        # Create comprehensive plots
                        fig_comprehensive = make_subplots(
                            rows=2, cols=2,
                            subplot_titles=('🏃 Vitesse vs Temps', '🚀 Accélération vs Temps', 
                                           '⚡ Puissance vs Temps', '🔋 Énergie Cinétique vs Temps'),
                            vertical_spacing=0.12
                        )
                        
                        # Velocity plot
                        fig_comprehensive.add_trace(
                            go.Scatter(x=metrics['time'], y=metrics['velocity']*1000, 
                                     mode='lines', name='Vitesse', line=dict(color='blue', width=2)),
                            row=1, col=1
                        )
                        
                        # Acceleration plot  
                        fig_comprehensive.add_trace(
                            go.Scatter(x=metrics['time'], y=metrics['acceleration']*1000,
                                     mode='lines', name='Accélération', line=dict(color='red', width=2)),
                            row=1, col=2
                        )
                        
                        # Power plot
                        fig_comprehensive.add_trace(
                            go.Scatter(x=metrics['time'], y=metrics['power']*1000,
                                     mode='lines', name='Puissance', line=dict(color='green', width=2)),
                            row=2, col=1
                        )
                        
                        # Energy plot
                        fig_comprehensive.add_trace(
                            go.Scatter(x=metrics['time'], y=metrics['energy_kinetic']*1000,
                                     mode='lines', name='Énergie', line=dict(color='purple', width=2)),
                            row=2, col=2
                        )
                        
                        # Update axes labels
                        fig_comprehensive.update_xaxes(title_text="Temps (s)")
                        fig_comprehensive.update_yaxes(title_text="Vitesse (mm/s)", row=1, col=1)
                        fig_comprehensive.update_yaxes(title_text="Accélération (mm/s²)", row=1, col=2)
                        fig_comprehensive.update_yaxes(title_text="Puissance (mW)", row=2, col=1)
                        fig_comprehensive.update_yaxes(title_text="Énergie (mJ)", row=2, col=2)
                        
                        fig_comprehensive.update_layout(height=600, showlegend=False)
                        st.plotly_chart(fig_comprehensive, use_container_width=True)
                        
                        # Physical interpretation
                        st.markdown("#### 🧠 Interprétation Physique")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("**🎯 Qualité de l'Expérience:**")
                            if metrics['trajectory_efficiency'] > 90:
                                st.success("✅ Trajectoire très droite - Excellente qualité")
                            elif metrics['trajectory_efficiency'] > 80:
                                st.success("✅ Trajectoire droite - Bonne qualité")
                            elif metrics['trajectory_efficiency'] > 70:
                                st.warning("⚠️ Trajectoire légèrement déviée")
                            else:
                                st.error("❌ Trajectoire très déviée - Vérifier le setup")
                            
                            if metrics['energy_efficiency'] > 70:
                                st.success("✅ Bonne conservation d'énergie")
                            elif metrics['energy_efficiency'] > 50:
                                st.warning("⚠️ Perte d'énergie modérée")
                            else:
                                st.error("❌ Perte d'énergie importante")
                        
                        with col2:
                            st.markdown("**📚 Comparaison Littérature:**")
                            if metrics['krr'] is not None:
                                if 0.03 <= metrics['krr'] <= 0.10:
                                    st.success("✅ Krr cohérent avec Van Wal (2017)")
                                elif metrics['krr'] < 0:
                                    st.error("⚠️ Krr négatif - sphère accélère!")
                                elif metrics['krr'] > 0.15:
                                    st.warning("⚠️ Krr très élevé - vérifier conditions")
                                else:
                                    st.info("💡 Krr en dehors de la gamme littérature")
                                
                                st.metric("Référence Van Wal", "0.05-0.07", f"{(metrics['krr']-0.06)/0.06*100:+.1f}%")
                        
                        # Export detailed data
                        st.markdown("#### 💾 Exporter les Données")
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            # Basic results
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
                        
                        with col2:
                            # Detailed time series
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
                        
                        with col3:
                            # Raw trajectory data
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
                        st.error("❌ Impossible de calculer Krr - données insuffisantes ou problème dans les calculs")
                        st.info("💡 Vérifiez que :")
                        st.info("• Vous avez au moins 10 détections valides")
                        st.info("• La sphère se déplace effectivement")
                        st.info("• Les paramètres physiques sont corrects")
            
            else:
                st.error("❌ Aucune donnée valide pour l'analyse avancée")
        
        # === DATA OVERVIEW ===
        else:  # Data overview
            st.markdown("""
            <div class="analysis-card">
                <h2>📋 Vue d'ensemble de vos données</h2>
                <p>Exploration et validation de la qualité des données</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Display first rows
            st.markdown("### 📊 Aperçu des Données")
            st.dataframe(df.head(10))
            
            # Descriptive statistics
            st.markdown("### 📈 Statistiques Descriptives")
            st.dataframe(df_valid.describe())
            
            # Distribution plots
            if len(df_valid) > 0:
                col1, col2 = st.columns(2)
                
                with col1:
                    fig1 = px.histogram(df_valid, x='Radius', title="Distribution des Rayons")
                    st.plotly_chart(fig1, use_container_width=True)
                    
                with col2:
                    detection_status = df['X_center'] != 0
                    fig2 = px.pie(values=[detection_status.sum(), (~detection_status).sum()],
                                 names=['Détecté', 'Non détecté'],
                                 title="Répartition des Détections")
                    st.plotly_chart(fig2, use_container_width=True)

    else:
        # Message if no data is loaded
        st.markdown("""
        ## 🚀 Pour commencer:
        
        1. **📂 Téléchargez votre fichier CSV** avec vos données expérimentales
        2. **Ou cliquez sur "Générer des données d'exemple"** pour explorer les fonctionnalités
        3. **🔧 Choisissez l'analyse** qui vous intéresse dans le menu
        
        ### 📋 Format de fichier attendu:
        Votre CSV doit contenir les colonnes suivantes:
        - `Frame`: Numéro d'image
        - `X_center`: Position X du centre de la sphère
        - `Y_center`: Position Y du centre de la sphère  
        - `Radius`: Rayon détecté de la sphère
        
        ### 🔧 Les 3 Analyses Détaillées Disponibles:
        - **Code 1**: Visualisation de trajectoire et qualité de détection
        - **Code 2**: Analyse Krr (coefficient de résistance) avec comparaison littérature
        - **Code 3**: Analyse complète et avancée avec métriques physiques
        """)

# ==================== MULTI-EXPERIMENT COMPARISON PAGE ====================
elif page == "🔍 Comparaison Multi-Expériences":
    
    st.markdown("""
    # 🔍 Comparaison Multi-Expériences
    ## Comparez plusieurs expériences pour analyser l'effet de différents paramètres
    """)
    
    # Check if experiments are available
    if not st.session_state.experiments:
        st.warning("⚠️ Aucune expérience disponible pour comparaison. Veuillez charger des expériences depuis la page d'analyse unique d'abord.")
        
        # Quick load sample experiments
        st.markdown("### 🚀 Démarrage Rapide: Charger des Expériences d'Exemple")
        if st.button("📊 Charger des expériences d'exemple pour comparaison"):
            # Create sample experiments with different water contents
            water_contents = [0, 5, 10, 15, 20]
            for w in water_contents:
                df, metadata = create_sample_data_with_metadata(f"Sample_W{w}%", w, "Steel")
                st.session_state.experiments[f"Sample_W{w}%"] = {
                    'data': df,
                    'metadata': metadata
                }
            st.success("✅ Expériences d'exemple chargées!")
            st.rerun()
    
    else:
        # Display available experiments
        st.markdown("### 📋 Expériences Disponibles")
        
        # Experiments overview table
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
        
        # Experiment selection for comparison
        st.markdown("### 🔬 Sélectionner les Expériences à Comparer")
        selected_experiments = st.multiselect(
            "Choisissez les expériences pour comparaison:",
            options=list(st.session_state.experiments.keys()),
            default=list(st.session_state.experiments.keys())[:min(4, len(st.session_state.experiments))]
        )
        
        if len(selected_experiments) >= 2:
            # Calculate comparison metrics
            comparison_data = []
            trajectory_data = []
            
            for exp_name in selected_experiments:
                exp = st.session_state.experiments[exp_name]
                df = exp['data']
                meta = exp['metadata']
                df_valid = df[(df['X_center'] != 0) & (df['Y_center'] != 0) & (df['Radius'] != 0)]
                
                # Calculate advanced metrics
                advanced_results = calculate_advanced_metrics(df_valid)
                
                comparison_data.append({
                    'Expérience': exp_name,
                    'Water_Content': meta['water_content'],
                    'Sphere_Type': meta['sphere_type'],
                    'Success_Rate': meta['success_rate'],
                    
                    # Basic metrics
                    'Krr': advanced_results['krr'] if advanced_results else None,
                    'Initial_Velocity': advanced_results['v0'] if advanced_results else None,
                    'Final_Velocity': advanced_results['vf'] if advanced_results else None,
                    'Distance': advanced_results['distance'] if advanced_results else None,
                    
                    # Advanced kinematic metrics
                    'Max_Velocity': advanced_results['max_velocity'] if advanced_results else None,
                    'Max_Acceleration': advanced_results['max_acceleration'] if advanced_results else None,
                    'Initial_Acceleration': advanced_results['initial_acceleration'] if advanced_results else None,
                    
                    # Force and energy metrics
                    'Max_Resistance_Force': advanced_results['max_resistance_force'] if advanced_results else None,
                    'Avg_Resistance_Force': advanced_results['avg_resistance_force'] if advanced_results else None,
                    'Max_Power': advanced_results['max_power'] if advanced_results else None,
                    'Energy_Dissipated': advanced_results['energy_dissipated'] if advanced_results else None,
                    'Energy_Efficiency': advanced_results['energy_efficiency'] if advanced_results else None,
                    
                    # Quality metrics
                    'Trajectory_Efficiency': advanced_results['trajectory_efficiency'] if advanced_results else None,
                    'Vertical_Variation': advanced_results['vertical_variation'] if advanced_results else None,
                    
                    # Time series for plotting
                    'time_series': advanced_results if advanced_results else None
                })
                
                # Trajectory data for overlay
                if len(df_valid) > 0:
                    df_traj = df_valid.copy()
                    df_traj['Experiment'] = exp_name
                    df_traj['Water_Content'] = meta['water_content']
                    trajectory_data.append(df_traj)
            
            comp_df = pd.DataFrame(comparison_data)
            
            # ===== ADVANCED COMPARISON VISUALIZATIONS =====
            st.markdown("### 📊 Analyse Comparative Avancée")
            
            # Create tabs for different types of analysis
            tab1, tab2, tab3, tab4 = st.tabs(["🏃 Analyse Cinématique", "⚡ Force & Énergie", "🛤️ Qualité Trajectoire", "📈 Séries Temporelles"])
            
            with tab1:
                st.markdown("#### 🏃 Comparaison Cinématique")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Acceleration comparison
                    if comp_df['Max_Acceleration'].notna().any():
                        fig_accel = px.scatter(comp_df, x='Water_Content', y='Max_Acceleration', 
                                             color='Sphere_Type', size='Success_Rate',
                                             hover_data=['Expérience'],
                                             title="🚀 Accélération Maximale vs Teneur en Eau",
                                             labels={'Max_Acceleration': 'Accélération Max (m/s²)', 
                                                    'Water_Content': 'Teneur en Eau (%)'})
                        st.plotly_chart(fig_accel, use_container_width=True)
                
                with col2:
                    # Velocity comparison enhanced
                    if comp_df['Max_Velocity'].notna().any():
                        fig_vel_max = px.bar(comp_df, x='Expérience', y='Max_Velocity',
                                           color='Water_Content',
                                           title="🏃 Comparaison Vitesse Maximale",
                                           labels={'Max_Velocity': 'Vitesse Max (m/s)'})
                        fig_vel_max.update_xaxes(tickangle=45)
                        st.plotly_chart(fig_vel_max, use_container_width=True)
                
                # Krr vs Water Content trend
                col1, col2 = st.columns(2)
                
                with col1:
                    if comp_df['Krr'].notna().any():
                        fig_krr = px.scatter(comp_df, x='Water_Content', y='Krr', 
                                           color='Sphere_Type', size='Success_Rate',
                                           hover_data=['Expérience'],
                                           title="🔍 Krr vs Teneur en Eau",
                                           labels={'Krr': 'Coefficient Krr', 
                                                  'Water_Content': 'Teneur en Eau (%)'})
                        
                        # Add trend line if enough points
                        if len(comp_df[comp_df['Krr'].notna()]) >= 3:
                            fig_krr.add_trace(go.Scatter(
                                x=comp_df['Water_Content'], 
                                y=comp_df['Krr'],
                                mode='lines',
                                name='Tendance',
                                line=dict(dash='dash', color='red')
                            ))
                        st.plotly_chart(fig_krr, use_container_width=True)
                
                with col2:
                    # Initial vs Final velocity comparison
                    if comp_df['Initial_Velocity'].notna().any() and comp_df['Final_Velocity'].notna().any():
                        fig_vel_comp = go.Figure()
                        
                        fig_vel_comp.add_trace(go.Scatter(
                            x=comp_df['Water_Content'],
                            y=comp_df['Initial_Velocity'] * 1000,  # Convert to mm/s
                            mode='markers+lines',
                            name='Vitesse Initiale',
                            marker=dict(color='blue', size=10)
                        ))
                        
                        fig_vel_comp.add_trace(go.Scatter(
                            x=comp_df['Water_Content'],
                            y=comp_df['Final_Velocity'] * 1000,  # Convert to mm/s
                            mode='markers+lines',
                            name='Vitesse Finale',
                            marker=dict(color='red', size=10)
                        ))
                        
                        fig_vel_comp.update_layout(
                            title="🏃 Comparaison Vitesses Initiale/Finale",
                            xaxis_title="Teneur en Eau (%)",
                            yaxis_title="Vitesse (mm/s)"
                        )
                        st.plotly_chart(fig_vel_comp, use_container_width=True)
            
            with tab2:
                st.markdown("#### ⚡ Analyse Force & Énergie")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Resistance force comparison
                    if comp_df['Max_Resistance_Force'].notna().any():
                        fig_force = px.scatter(comp_df, x='Water_Content', y='Max_Resistance_Force',
                                             color='Sphere_Type', size='Success_Rate',
                                             hover_data=['Expérience'],
                                             title="🔧 Force de Résistance Max vs Teneur en Eau",
                                             labels={'Max_Resistance_Force': 'Force Résistance Max (N)',
                                                    'Water_Content': 'Teneur en Eau (%)'})
                        st.plotly_chart(fig_force, use_container_width=True)
                
                with col2:
                    # Power comparison
                    if comp_df['Max_Power'].notna().any():
                        fig_power = px.bar(comp_df, x='Expérience', y='Max_Power',
                                         color='Water_Content',
                                         title="⚡ Puissance Maximale Dissipée",
                                         labels={'Max_Power': 'Puissance Max (W)'})
                        fig_power.update_xaxes(tickangle=45)
                        st.plotly_chart(fig_power, use_container_width=True)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Energy dissipated
                    if comp_df['Energy_Dissipated'].notna().any():
                        fig_energy = px.scatter(comp_df, x='Water_Content', y='Energy_Dissipated',
                                              color='Sphere_Type',
                                              title="🔋 Énergie Dissipée vs Teneur en Eau",
                                              labels={'Energy_Dissipated': 'Énergie Dissipée (J)',
                                                     'Water_Content': 'Teneur en Eau (%)'})
                        st.plotly_chart(fig_energy, use_container_width=True)
                
                with col2:
                    # Energy efficiency
                    if comp_df['Energy_Efficiency'].notna().any():
                        fig_efficiency = px.bar(comp_df, x='Expérience', y='Energy_Efficiency',
                                              color='Water_Content',
                                              title="🎯 Efficacité Énergétique",
                                              labels={'Energy_Efficiency': 'Efficacité Énergétique (%)'})
                        fig_efficiency.update_xaxes(tickangle=45)
                        st.plotly_chart(fig_efficiency, use_container_width=True)
            
            with tab3:
                st.markdown("#### 🛤️ Analyse Qualité Trajectoire")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Trajectory efficiency
                    if comp_df['Trajectory_Efficiency'].notna().any():
                        fig_traj_eff = px.bar(comp_df, x='Expérience', y='Trajectory_Efficiency',
                                            color='Water_Content',
                                            title="📐 Efficacité de Trajectoire (Rectitude)",
                                            labels={'Trajectory_Efficiency': 'Efficacité Trajectoire (%)'})
                        fig_traj_eff.update_xaxes(tickangle=45)
                        st.plotly_chart(fig_traj_eff, use_container_width=True)
                
                with col2:
                    # Vertical variation
                    if comp_df['Vertical_Variation'].notna().any():
                        fig_vert_var = px.scatter(comp_df, x='Water_Content', y='Vertical_Variation',
                                                color='Sphere_Type',
                                                title="📏 Variation Verticale du Chemin",
                                                labels={'Vertical_Variation': 'Variation Verticale (mm)',
                                                       'Water_Content': 'Teneur en Eau (%)'})
                        st.plotly_chart(fig_vert_var, use_container_width=True)
                
                # Combined trajectory overlay (enhanced)
                if trajectory_data:
                    st.markdown("##### 🎯 Comparaison de Trajectoires Améliorée")
                    
                    combined_traj = pd.concat(trajectory_data, ignore_index=True)
                    
                    # Create trajectory plot with quality metrics
                    fig_traj_enhanced = px.scatter(combined_traj, x='X_center', y='Y_center',
                                                 color='Experiment', 
                                                 animation_frame='Frame',
                                                 title="🛤️ Comparaison Trajectoires avec Métriques Qualité",
                                                 opacity=0.7)
                    fig_traj_enhanced.update_yaxes(autorange="reversed")
                    fig_traj_enhanced.update_layout(height=600)
                    st.plotly_chart(fig_traj_enhanced, use_container_width=True)
            
            with tab4:
                st.markdown("#### 📈 Comparaison Séries Temporelles")
                
                # Create time series plots for selected experiments
                if len(selected_experiments) <= 5:  # Limit to avoid cluttered plots
                    
                    # Velocity profiles
                    st.markdown("##### 🏃 Profils de Vitesse")
                    fig_vel_series = go.Figure()
                    
                    for exp_name in selected_experiments:
                        exp_data = next((item for item in comparison_data if item['Expérience'] == exp_name), None)
                        if exp_data and exp_data['time_series']:
                            ts = exp_data['time_series']
                            fig_vel_series.add_trace(go.Scatter(
                                x=ts['time'], 
                                y=ts['velocity'] * 1000,  # Convert to mm/s
                                mode='lines',
                                name=f"{exp_name} ({exp_data['Water_Content']}% eau)",
                                line=dict(width=2)
                            ))
                    
                    fig_vel_series.update_layout(
                        title="Vitesse vs Temps - Toutes Expériences",
                        xaxis_title="Temps (s)",
                        yaxis_title="Vitesse (mm/s)",
                        height=400
                    )
                    st.plotly_chart(fig_vel_series, use_container_width=True)
                    
                    # Acceleration profiles
                    st.markdown("##### 🚀 Profils d'Accélération")
                    fig_accel_series = go.Figure()
                    
                    for exp_name in selected_experiments:
                        exp_data = next((item for item in comparison_data if item['Expérience'] == exp_name), None)
                        if exp_data and exp_data['time_series']:
                            ts = exp_data['time_series']
                            fig_accel_series.add_trace(go.Scatter(
                                x=ts['time'], 
                                y=ts['acceleration'] * 1000,  # Convert to mm/s²
                                mode='lines',
                                name=f"{exp_name} ({exp_data['Water_Content']}% eau)",
                                line=dict(width=2)
                            ))
                    
                    fig_accel_series.update_layout(
                        title="Accélération vs Temps - Toutes Expériences",
                        xaxis_title="Temps (s)",
                        yaxis_title="Accélération (mm/s²)",
                        height=400
                    )
                    st.plotly_chart(fig_accel_series, use_container_width=True)
                    
                    # Power profiles
                    st.markdown("##### ⚡ Profils de Dissipation de Puissance")
                    fig_power_series = go.Figure()
                    
                    for exp_name in selected_experiments:
                        exp_data = next((item for item in comparison_data if item['Expérience'] == exp_name), None)
                        if exp_data and exp_data['time_series']:
                            ts = exp_data['time_series']
                            fig_power_series.add_trace(go.Scatter(
                                x=ts['time'], 
                                y=ts['power'] * 1000,  # Convert to mW
                                mode='lines',
                                name=f"{exp_name} ({exp_data['Water_Content']}% eau)",
                                line=dict(width=2)
                            ))
                    
                    fig_power_series.update_layout(
                        title="Dissipation de Puissance vs Temps - Toutes Expériences",
                        xaxis_title="Temps (s)",
                        yaxis_title="Puissance (mW)",
                        height=400
                    )
                    st.plotly_chart(fig_power_series, use_container_width=True)
                    
                else:
                    st.warning("⚠️ Trop d'expériences sélectionnées pour la comparaison des séries temporelles. Veuillez sélectionner 5 expériences ou moins.")
            
            # Enhanced statistical comparison table
            st.markdown("### 📋 Tableau de Comparaison Détaillé")
            
            # Format the comparison table with all new metrics
            display_comp = comp_df.copy()
            
            # Format numeric columns
            numeric_columns = {
                'Krr': lambda x: f"{x:.6f}" if pd.notna(x) else "N/A",
                'Initial_Velocity': lambda x: f"{x*1000:.2f}" if pd.notna(x) else "N/A",
                'Final_Velocity': lambda x: f"{x*1000:.2f}" if pd.notna(x) else "N/A",
                'Max_Velocity': lambda x: f"{x*1000:.2f}" if pd.notna(x) else "N/A",
                'Distance': lambda x: f"{x*1000:.2f}" if pd.notna(x) else "N/A",
                'Max_Acceleration': lambda x: f"{x*1000:.2f}" if pd.notna(x) else "N/A",
                'Max_Resistance_Force': lambda x: f"{x*1000:.2f}" if pd.notna(x) else "N/A",
                'Energy_Dissipated': lambda x: f"{x*1000:.2f}" if pd.notna(x) else "N/A",
                'Energy_Efficiency': lambda x: f"{x:.1f}" if pd.notna(x) else "N/A",
                'Trajectory_Efficiency': lambda x: f"{x:.1f}" if pd.notna(x) else "N/A"
            }
            
            for col, formatter in numeric_columns.items():
                if col in display_comp.columns:
                    new_col_name = col.replace('_', ' ') + (' (mm/s)' if 'Velocity' in col else 
                                                           ' (mm)' if col == 'Distance' else
                                                           ' (mm/s²)' if 'Acceleration' in col else
                                                           ' (mN)' if 'Force' in col else
                                                           ' (mJ)' if 'Energy_Dissipated' in col else
                                                           ' (%)' if 'Efficiency' in col else '')
                    display_comp[new_col_name] = display_comp[col].apply(formatter)
            
            # Select relevant columns for display
            display_columns = ['Expérience', 'Water_Content', 'Sphere_Type', 'Success_Rate', 
                             'Krr', 'Max Velocity (mm/s)', 'Max Acceleration (mm/s²)', 
                             'Max Resistance Force (mN)', 'Energy Dissipated (mJ)', 
                             'Energy Efficiency (%)', 'Trajectory Efficiency (%)']
            
            available_columns = [col for col in display_columns if col in display_comp.columns]
            st.dataframe(display_comp[available_columns], use_container_width=True)
            
            # Enhanced key insights
            st.markdown("### 🔍 Insights Clés Avancés")
            
            if len(comp_df) >= 2:
                # Create insights based on new metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    if comp_df['Krr'].notna().sum() >= 2:
                        water_sorted = comp_df.sort_values('Water_Content')
                        krr_change = water_sorted['Krr'].iloc[-1] - water_sorted['Krr'].iloc[0]
                        water_change = water_sorted['Water_Content'].iloc[-1] - water_sorted['Water_Content'].iloc[0]
                        
                        st.markdown(f"""
                        <div class="comparison-card">
                            <h4>💧 Effet Teneur en Eau</h4>
                            <p>Variation Krr: <strong>{krr_change:.6f}</strong></p>
                            <p>Gamme eau: <strong>{water_change:.1f}%</strong></p>
                        </div>
                        """, unsafe_allow_html=True)
                
                with col2:
                    if comp_df['Max_Resistance_Force'].notna().any():
                        max_force_exp = comp_df.loc[comp_df['Max_Resistance_Force'].idxmax()]
                        st.markdown(f"""
                        <div class="comparison-card">
                            <h4>🔧 Résistance Maximale</h4>
                            <p><strong>{max_force_exp['Expérience']}</strong></p>
                            <p>{max_force_exp['Max_Resistance_Force']*1000:.1f} mN</p>
                        </div>
                        """, unsafe_allow_html=True)
                
                with col3:
                    if comp_df['Energy_Efficiency'].notna().any():
                        best_efficiency_exp = comp_df.loc[comp_df['Energy_Efficiency'].idxmax()]
                        st.markdown(f"""
                        <div class="comparison-card">
                            <h4>⚡ Meilleure Efficacité Énergétique</h4>
                            <p><strong>{best_efficiency_exp['Expérience']}</strong></p>
                            <p>{best_efficiency_exp['Energy_Efficiency']:.1f}% conservée</p>
                        </div>
                        """, unsafe_allow_html=True)
                
                with col4:
                    if comp_df['Trajectory_Efficiency'].notna().any():
                        best_trajectory_exp = comp_df.loc[comp_df['Trajectory_Efficiency'].idxmax()]
                        st.markdown(f"""
                        <div class="comparison-card">
                            <h4>🛤️ Trajectoire la Plus Droite</h4>
                            <p><strong>{best_trajectory_exp['Expérience']}</strong></p>
                            <p>{best_trajectory_exp['Trajectory_Efficiency']:.1f}% efficacité</p>
                        </div>
                        """, unsafe_allow_html=True)
                
                # Correlation analysis
                st.markdown("#### 🔗 Analyse de Corrélation Avancée")
                
                # Select numeric columns for correlation
                correlation_columns = ['Water_Content', 'Krr', 'Max_Acceleration', 'Max_Resistance_Force', 
                                     'Energy_Dissipated', 'Energy_Efficiency', 'Trajectory_Efficiency']
                
                available_corr_columns = [col for col in correlation_columns if col in comp_df.columns and comp_df[col].notna().any()]
                
                if len(available_corr_columns) >= 3:
                    corr_matrix = comp_df[available_corr_columns].corr()
                    
                    fig_corr = px.imshow(
                        corr_matrix, 
                        text_auto=True, 
                        aspect="auto",
                        title="📊 Matrice de Corrélation - Tous Paramètres",
                        color_continuous_scale="RdBu_r"
                    )
                    fig_corr.update_layout(height=500)
                    st.plotly_chart(fig_corr, use_container_width=True)
                    
                    # Key correlations insights
                    st.markdown("##### 🎯 Corrélations Clés Trouvées:")
                    
                    # Find strongest correlations (excluding self-correlations)
                    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
                    corr_values = corr_matrix.where(mask).stack().reset_index()
                    corr_values.columns = ['Var1', 'Var2', 'Correlation']
                    corr_values = corr_values.sort_values('Correlation', key=abs, ascending=False)
                    
                    for i, row in corr_values.head(3).iterrows():
                        correlation_strength = "Forte" if abs(row['Correlation']) > 0.7 else "Modérée" if abs(row['Correlation']) > 0.5 else "Faible"
                        correlation_direction = "positive" if row['Correlation'] > 0 else "négative"
                        
                        st.markdown(f"- **Corrélation {correlation_strength} {correlation_direction}** entre {row['Var1']} et {row['Var2']} (r = {row['Correlation']:.3f})")
                
                # Physical insights section
                st.markdown("#### 🧠 Insights Physiques")
                
                insights = []
                
                # Water content effect on Krr
                if 'Water_Content' in comp_df.columns and 'Krr' in comp_df.columns:
                    water_krr_corr = comp_df[['Water_Content', 'Krr']].corr().iloc[0, 1]
                    if not pd.isna(water_krr_corr):
                        if water_krr_corr > 0.3:
                            insights.append("💧 **L'humidité augmente la résistance au roulement** - le substrat humide crée plus de traînée")
                        elif water_krr_corr < -0.3:
                            insights.append("💧 **L'humidité diminue la résistance au roulement** - possible effet de lubrification")
                        else:
                            insights.append("💧 **L'humidité a un effet minimal** sur la résistance au roulement dans cette gamme")
                
                # Force vs acceleration relationship
                if 'Max_Acceleration' in comp_df.columns and 'Max_Resistance_Force' in comp_df.columns:
                    force_accel_corr = comp_df[['Max_Acceleration', 'Max_Resistance_Force']].corr().iloc[0, 1]
                    if not pd.isna(force_accel_corr) and force_accel_corr > 0.7:
                        insights.append("🔧 **Couplage force-accélération fort** - Deuxième loi de Newton bien vérifiée")
                
                # Energy efficiency insights
                if 'Energy_Efficiency' in comp_df.columns and 'Water_Content' in comp_df.columns:
                    energy_water_corr = comp_df[['Energy_Efficiency', 'Water_Content']].corr().iloc[0, 1]
                    if not pd.isna(energy_water_corr):
                        if energy_water_corr < -0.3:
                            insights.append("⚡ **Perte d'énergie augmente avec l'humidité** - plus de processus dissipatifs")
                        elif energy_water_corr > 0.3:
                            insights.append("⚡ **Efficacité énergétique s'améliore avec l'humidité** - lubrification inattendue?")
                
                # Trajectory quality insights
                if 'Trajectory_Efficiency' in comp_df.columns and 'Water_Content' in comp_df.columns:
                    traj_water_corr = comp_df[['Trajectory_Efficiency', 'Water_Content']].corr().iloc[0, 1]
                    if not pd.isna(traj_water_corr):
                        if traj_water_corr < -0.3:
                            insights.append("🛤️ **Le chemin devient moins droit avec l'humidité** - résistance latérale accrue")
                        elif traj_water_corr > 0.3:
                            insights.append("🛤️ **Le chemin devient plus droit avec l'humidité** - déflexion latérale réduite")
                
                if insights:
                    for insight in insights:
                        st.markdown(insight)
                else:
                    st.markdown("📊 *Plus d'expériences nécessaires pour des insights physiques robustes*")
            
            # Export enhanced comparison results
            st.markdown("### 💾 Exporter les Résultats Améliorés")
            
            # Create comprehensive export data
            export_data = comp_df.copy()
            
            # Add summary statistics
            summary_stats = {}
            numeric_cols = export_data.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if export_data[col].notna().any():
                    summary_stats[f"{col}_mean"] = export_data[col].mean()
                    summary_stats[f"{col}_std"] = export_data[col].std()
                    summary_stats[f"{col}_min"] = export_data[col].min()
                    summary_stats[f"{col}_max"] = export_data[col].max()
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                csv_comparison = export_data.to_csv(index=False)
                st.download_button(
                    label="📥 Télécharger données de comparaison (CSV)",
                    data=csv_comparison,
                    file_name="comparaison_experiences_avancee.csv",
                    mime="text/csv"
                )
            
            with col2:
                # Export summary statistics
                summary_df = pd.DataFrame([summary_stats])
                csv_summary = summary_df.to_csv(index=False)
                st.download_button(
                    label="📊 Télécharger statistiques résumé (CSV)",
                    data=csv_summary,
                    file_name="statistiques_resume_experiences.csv",
                    mime="text/csv"
                )
            
            with col3:
                # Export correlation matrix
                if len(available_corr_columns) >= 3:
                    csv_correlation = corr_matrix.to_csv()
                    st.download_button(
                        label="🔗 Télécharger matrice corrélation (CSV)",
                        data=csv_correlation,
                        file_name="matrice_correlation.csv",
                        mime="text/csv"
                    )
        
        else:
            st.info("Veuillez sélectionner au moins 2 expériences pour la comparaison")
        
        # Experiment management
        st.markdown("### 🗂️ Gestion des Expériences")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Supprimer une Expérience:**")
            exp_to_remove = st.selectbox("Sélectionner l'expérience à supprimer:", 
                                       options=["Aucune"] + list(st.session_state.experiments.keys()))
            
            if exp_to_remove != "Aucune" and st.button("🗑️ Supprimer l'Expérience Sélectionnée"):
                del st.session_state.experiments[exp_to_remove]
                st.success(f"Expérience '{exp_to_remove}' supprimée!")
                st.rerun()
        
        with col2:
            st.markdown("**Tout Effacer:**")
            st.write("⚠️ Ceci supprimera toutes les expériences sauvegardées")
            if st.button("🧹 Effacer Toutes les Expériences"):
                st.session_state.experiments = {}
                st.success("Toutes les expériences effacées!")
                st.rerun()

# ==================== PREDICTION MODULE PAGE ====================
elif page == "🎯 Module de Prédiction":
    
    st.markdown("""
    # 🎯 Module de Prédiction
    ## Assistant Prédictif d'Ingénierie pour la Résistance au Roulement
    """)
    
    if not st.session_state.experiments:
        st.warning("⚠️ Aucune donnée expérimentale disponible pour les prédictions. Veuillez charger des expériences depuis la page d'analyse unique d'abord.")
        
        if st.button("📊 Charger des données d'exemple pour la démo de prédiction"):
            # Create sample experiments
            water_contents = [0, 5, 10, 15, 20, 25]
            for w in water_contents:
                df, metadata = create_sample_data_with_metadata(f"Sample_W{w}%", w, "Steel")
                st.session_state.experiments[f"Sample_W{w}%"] = {
                    'data': df,
                    'metadata': metadata
                }
            st.success("✅ Données d'exemple chargées pour la prédiction!")
            st.rerun()
    
    else:
        # Build prediction models
        models = build_prediction_model(st.session_state.experiments)
        
        if not models:
            st.error("❌ Données insuffisantes pour construire des modèles de prédiction fiables. Besoin d'au moins 3 expériences avec des résultats valides.")
        else:
            st.success(f"✅ Modèles de prédiction construits à partir de {len(st.session_state.experiments)} expériences!")
            
            # Model quality overview
            st.markdown("### 📊 Évaluation de la Qualité des Modèles")
            
            col1, col2, col3 = st.columns(3)
            
            for i, (param, model) in enumerate(models.items()):
                param_name = param.replace('_', ' ').title()
                
                with [col1, col2, col3][i % 3]:
                    r2_score = model['r2']
                    quality = "Excellent" if r2_score > 0.8 else "Bon" if r2_score > 0.6 else "Modéré" if r2_score > 0.4 else "Faible"
                    color = "🟢" if r2_score > 0.8 else "🟡" if r2_score > 0.6 else "🟠" if r2_score > 0.4 else "🔴"
                    
                    st.markdown(f"""
                    <div class="metric-card">
                        <h4>{color} {param_name}</h4>
                        <p>R² = {r2_score:.3f}</p>
                        <p>{quality} Modèle</p>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Prediction interface
            st.markdown("### 🔮 Faire des Prédictions")
            
            col1, col2 = st.columns([2, 3])
            
            with col1:
                st.markdown("#### Conditions d'Entrée")
                
                # Get data ranges for validation
                all_water = [exp['metadata']['water_content'] for exp in st.session_state.experiments.values()]
                min_water, max_water = min(all_water), max(all_water)
                
                pred_water = st.slider(
                    "Teneur en Eau (%)", 
                    min_value=0.0, 
                    max_value=30.0, 
                    value=10.0, 
                    step=0.5,
                    help=f"Modèle entraîné sur la gamme {min_water}%-{max_water}%"
                )
                
                sphere_materials = list(set([exp['metadata']['sphere_type'] for exp in st.session_state.experiments.values()]))
                if len(sphere_materials) > 1:
                    pred_material = st.selectbox("Matériau de la Sphère", sphere_materials)
                else:
                    pred_material = sphere_materials[0]
                    st.info(f"Utilisation {pred_material} (seul matériau dans le dataset)")
                
                confidence_level = st.selectbox("Niveau de Confiance", [90, 95], index=1)
                
                # Advanced prediction options
                with st.expander("🔧 Options Avancées"):
                    show_equations = st.checkbox("Afficher les équations de prédiction", value=False)
                    explain_extrapolation = st.checkbox("Expliquer les avertissements d'extrapolation", value=True)
            
            with col2:
                st.markdown("#### Prédictions & Intervalles de Confiance")
                
                # Make predictions for each model
                predictions = {}
                
                for param, model in models.items():
                    pred, ci_lower, ci_upper, extrapolation = predict_with_confidence(
                        model, pred_water, confidence_level/100
                    )
                    
                    predictions[param] = {
                        'value': pred,
                        'ci_lower': ci_lower,
                        'ci_upper': ci_upper,
                        'extrapolation': extrapolation
                    }
                    
                    param_name = param.replace('_', ' ').title()
                    unit = ""
                    if param == 'krr':
                        unit = ""
                    elif 'velocity' in param:
                        unit = " mm/s"
                        pred *= 1000
                        ci_lower *= 1000
                        ci_upper *= 1000
                    elif 'efficiency' in param:
                        unit = "%"
                    elif 'force' in param:
                        unit = " mN"
                        pred *= 1000
                        ci_lower *= 1000
                        ci_upper *= 1000
                    
                    # Display prediction with confidence interval
                    extrap_warning = "⚠️ " if extrapolation else ""
                    
                    st.markdown(f"""
                    <div class="prediction-card">
                        <h4>{extrap_warning}{param_name}</h4>
                        <p><strong>Prédiction: {pred:.4f}{unit}</strong></p>
                        <p>IC {confidence_level}%: [{ci_lower:.4f}, {ci_upper:.4f}]{unit}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    if extrapolation and explain_extrapolation:
                        st.markdown(f"""
                        <div class="warning-card">
                            ⚠️ <strong>Extrapolation:</strong> {pred_water}% est en dehors de la gamme d'entraînement ({min_water}%-{max_water}%)
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Show equation if requested
                    if show_equations:
                        if model['degree'] == 2:
                            a, b, c = model['coeffs']
                            st.caption(f"📐 Équation: {param_name} = {a:.6f}×W² + {b:.6f}×W + {c:.6f}")
                        else:
                            a, b = model['coeffs']
                            st.caption(f"📐 Équation: {param_name} = {a:.6f}×W + {b:.6f}")
            
            # Prediction visualization
            st.markdown("### 📈 Visualisation des Prédictions")
            
            # Create prediction plots
            fig_predictions = make_subplots(
                rows=1, cols=len(models),
                subplot_titles=[param.replace('_', ' ').title() for param in models.keys()],
                horizontal_spacing=0.1
            )
            
            # Water content range for smooth curves
            water_range = np.linspace(0, 30, 100)
            
            for i, (param, model) in enumerate(models.items()):
                # Predict for the range
                predictions_range = [np.polyval(model['coeffs'], w) for w in water_range]
                
                # Add prediction curve
                fig_predictions.add_trace(
                    go.Scatter(x=water_range, y=predictions_range,
                             mode='lines', name=f'{param} prédiction',
                             line=dict(color='blue', width=2)),
                    row=1, col=i+1
                )
                
                # Add data points from experiments
                exp_water = []
                exp_values = []
                for exp_name, exp in st.session_state.experiments.items():
                    df = exp['data']
                    meta = exp['metadata']
                    df_valid = df[(df['X_center'] != 0) & (df['Y_center'] != 0) & (df['Radius'] != 0)]
                    
                    metrics = calculate_advanced_metrics(df_valid)
                    if metrics and param in metrics and metrics[param] is not None:
                        exp_water.append(meta['water_content'])
                        exp_values.append(metrics[param])
                
                if exp_water:
                    fig_predictions.add_trace(
                        go.Scatter(x=exp_water, y=exp_values,
                                 mode='markers', name=f'{param} données',
                                 marker=dict(color='red', size=8)),
                        row=1, col=i+1
                    )
                
                # Add current prediction point
                current_pred = predictions[param]['value']
                fig_predictions.add_trace(
                    go.Scatter(x=[pred_water], y=[current_pred],
                             mode='markers', name=f'{param} prédiction actuelle',
                             marker=dict(color='green', size=12, symbol='star')),
                    row=1, col=i+1
                )
                
                # Add confidence bands
                ci_lower_range = [predictions[param]['ci_lower']] * len(water_range)
                ci_upper_range = [predictions[param]['ci_upper']] * len(water_range)
                
                fig_predictions.add_trace(
                    go.Scatter(x=water_range, y=ci_upper_range,
                             mode='lines', line=dict(color='lightblue', dash='dash'),
                             name=f'{param} IC sup', showlegend=False),
                    row=1, col=i+1
                )
                
                fig_predictions.add_trace(
                    go.Scatter(x=water_range, y=ci_lower_range,
                             mode='lines', line=dict(color='lightblue', dash='dash'),
                             fill='tonexty', fillcolor='rgba(173,216,230,0.3)',
                             name=f'{param} IC inf', showlegend=False),
                    row=1, col=i+1
                )
            
            fig_predictions.update_layout(height=400, showlegend=False)
            fig_predictions.update_xaxes(title_text="Teneur en Eau (%)")
            st.plotly_chart(fig_predictions, use_container_width=True)
            
            # Engineering recommendations
            st.markdown("### 🔧 Recommandations d'Ingénierie")
            
            recommendations = generate_engineering_recommendations(st.session_state.experiments, models)
            
            for rec in recommendations:
                st.markdown(rec)

# ==================== AUTO-GENERATED REPORT PAGE ====================
elif page == "📊 Rapport Auto-Généré":
    
    st.markdown("""
    # 📊 Rapport d'Analyse Auto-Généré
    ## Résumé d'Analyse Complet & Recommandations
    """)
    
    if not st.session_state.experiments:
        st.warning("⚠️ Aucune donnée expérimentale disponible pour la génération de rapport. Veuillez charger des expériences d'abord.")
        
        if st.button("📊 Charger des données d'exemple pour la démo de rapport"):
            # Create comprehensive sample experiments
            conditions = [
                (0, "Steel"), (5, "Steel"), (10, "Steel"), (15, "Steel"), (20, "Steel"),
                (10, "Plastic"), (15, "Plastic")
            ]
            
            for water, material in conditions:
                df, metadata = create_sample_data_with_metadata(f"{material}_W{water}%", water, material)
                st.session_state.experiments[f"{material}_W{water}%"] = {
                    'data': df,
                    'metadata': metadata
                }
            st.success("✅ Données d'exemple complètes chargées!")
            st.rerun()
    
    else:
        # Generate comprehensive report
        with st.spinner("🔄 Génération du rapport d'analyse complet..."):
            
            # Calculate comprehensive statistics
            total_experiments = len(st.session_state.experiments)
            water_contents = [exp['metadata']['water_content'] for exp in st.session_state.experiments.values()]
            success_rates = [exp['metadata']['success_rate'] for exp in st.session_state.experiments.values()]
            
            # Calculate advanced metrics for all experiments
            all_metrics = []
            for exp_name, exp in st.session_state.experiments.items():
                df = exp['data']
                meta = exp['metadata']
                df_valid = df[(df['X_center'] != 0) & (df['Y_center'] != 0) & (df['Radius'] != 0)]
                
                metrics = calculate_advanced_metrics(df_valid)
                if metrics:
                    all_metrics.append({
                        'exp_name': exp_name,
                        'water_content': meta['water_content'],
                        'sphere_type': meta['sphere_type'],
                        **metrics
                    })
            
            # Build models for report
            models = build_prediction_model(st.session_state.experiments)
        
        # Display report
        st.markdown("### 📋 Rapport Généré")
        
        # Report controls
        col1, col2, col3 = st.columns(3)
        
        with col1:
            current_time = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # Generate text report
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
            
            report_content += "\n## 🔧 RECOMMANDATIONS D'INGÉNIERIE\n\n"
            recommendations = generate_engineering_recommendations(st.session_state.experiments, models)
            for rec in recommendations:
                report_content += f"{rec}\n"
            
            if models:
                report_content += "\n## 📈 MODÈLES PRÉDICTIFS\n\n"
                for param, model in models.items():
                    param_name = param.replace('_', ' ').title()
                    report_content += f"### {param_name}\n"
                    report_content += f"• **Qualité du modèle (R²)**: {model['r2']:.3f}\n"
                    report_content += f"• **Gamme valide**: {model['data_range'][0]:.1f}% - {model['data_range'][1]:.1f}% teneur en eau\n"
                    report_content += f"• **Erreur standard**: ±{model['std_error']:.6f}\n\n"
            
            # Download report as text
            st.download_button(
                label="📥 Télécharger le rapport (TXT)",
                data=report_content,
                file_name=f"rapport_analyse_{current_time}.txt",
                mime="text/plain"
            )
        
        with col2:
            # Generate comprehensive CSV data export
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
                        'Trajectory_Efficiency_percent': metric['trajectory_efficiency'],
                        'Max_Acceleration_mm_s2': metric['max_acceleration'] * 1000 if metric['max_acceleration'] else None,
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
        
        with col3:
            # Email report option (placeholder)
            if st.button("📧 Envoyer par email"):
                st.info("💡 Fonction email en développement. Utilisez l'option de téléchargement.")
        
        # Interactive dashboard
        st.markdown("---")
        st.markdown("### 📊 Tableau de Bord Interactif")
        
        # Summary statistics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Expériences Totales", total_experiments)
        
        with col2:
            water_range = max(water_contents) - min(water_contents)
            st.metric("Gamme d'Humidité", f"{water_range:.1f}%")
        
        with col3:
            st.metric("Succès Moyen", f"{np.mean(success_rates):.1f}%")
        
        with col4:
            model_count = len(models) if models else 0
            st.metric("Modèles Générés", model_count)
        
        # Quality assessment
        st.markdown("### 🎯 Évaluation de la Qualité")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Qualité des Données")
            avg_success = np.mean(success_rates)
            
            if avg_success >= 80:
                st.success("✅ Excellente qualité de détection")
                quality_score = "A+"
            elif avg_success >= 70:
                st.success("✅ Bonne qualité de détection")
                quality_score = "A"
            elif avg_success >= 60:
                st.warning("⚠️ Qualité de détection modérée")
                quality_score = "B"
            else:
                st.error("❌ Qualité de détection faible")
                quality_score = "C"
            
            st.metric("Score de Qualité", quality_score)
        
        with col2:
            st.markdown("#### Fiabilité des Modèles")
            
            if models:
                avg_r2 = np.mean([model['r2'] for model in models.values()])
                
                if avg_r2 >= 0.8:
                    st.success("✅ Modèles très fiables")
                    model_grade = "A+"
                elif avg_r2 >= 0.6:
                    st.success("✅ Modèles fiables")
                    model_grade = "A"
                elif avg_r2 >= 0.4:
                    st.warning("⚠️ Modèles modérément fiables")
                    model_grade = "B"
                else:
                    st.error("❌ Modèles peu fiables")
                    model_grade = "C"
                
                st.metric("R² Moyen", f"{avg_r2:.3f}")
                st.metric("Grade du Modèle", model_grade)
            else:
                st.warning("Aucun modèle disponible")
        
        # Comprehensive visualization summary
        st.markdown("### 📈 Résumé Visuel")
        
        if all_metrics:
            # Create summary plots
            summary_data = pd.DataFrame(all_metrics)
            
            col1, col2 = st.columns(2)
            
            with col1:
                if 'krr' in summary_data.columns and summary_data['krr'].notna().any():
                    fig_krr_summary = px.scatter(summary_data, x='water_content', y='krr',
                                               color='sphere_type',
                                               title="📊 Résumé: Krr vs Teneur en Eau",
                                               labels={'water_content': 'Teneur en Eau (%)',
                                                      'krr': 'Coefficient Krr'})
                    st.plotly_chart(fig_krr_summary, use_container_width=True)
            
            with col2:
                if 'energy_efficiency' in summary_data.columns and summary_data['energy_efficiency'].notna().any():
                    fig_energy_summary = px.bar(summary_data, x='exp_name', y='energy_efficiency',
                                               color='water_content',
                                               title="📊 Résumé: Efficacité Énergétique",
                                               labels={'exp_name': 'Expérience',
                                                      'energy_efficiency': 'Efficacité Énergétique (%)'})
                    fig_energy_summary.update_xaxes(tickangle=45)
                    st.plotly_chart(fig_energy_summary, use_container_width=True)
        
        # Recommendations for improvement
        st.markdown("### 💡 Recommandations d'Amélioration")
        
        improvement_recs = []
        
        if total_experiments < 5:
            improvement_recs.append("📊 **Augmenter le nombre d'expériences** - Collecter au moins 5-8 expériences pour des modèles robustes")
        
        if max(water_contents) - min(water_contents) < 15:
            improvement_recs.append("💧 **Élargir la gamme d'humidité** - Tester une gamme plus large de teneurs en eau")
        
        if avg_success < 75:
            improvement_recs.append("🔧 **Améliorer la qualité de détection** - Optimiser les paramètres de détection ou l'éclairage")
        
        sphere_types = set([exp['metadata']['sphere_type'] for exp in st.session_state.experiments.values()])
        if len(sphere_types) < 2:
            improvement_recs.append("⚪ **Tester différents matériaux** - Inclure plusieurs types de sphères pour la comparaison")
        
        if improvement_recs:
            for rec in improvement_recs:
                st.markdown(rec)
        else:
            st.success("✅ Configuration expérimentale excellente! Aucune amélioration majeure nécessaire.")

# Footer
st.markdown("---")
st.markdown("""
### 🎓 Plateforme d'Analyse de Résistance au Roulement des Sphères
*Développée pour analyser la résistance au roulement des sphères sur matériau granulaire humide*

**Institution**: Department of Cosmic Earth Science, Graduate School of Science, Osaka University  
**Domaine**: Mécanique granulaire  
**Innovation**: Première étude de l'effet de l'humidité  
**Applications**: Ingénierie géotechnique, systèmes de transport, mécanique des sols
""")

# Sidebar information
st.sidebar.markdown("---")
st.sidebar.markdown("""
### 📊 Statistiques du Projet
- **Images traitées**: 107
- **Taux de succès**: 76.6%
- **Méthode de détection**: Vision par ordinateur
- **Type de recherche**: Physique expérimentale
""")

st.sidebar.markdown("""
### 🎓 Contexte de Recherche
**Institution**: Université d'Osaka  
**Domaine**: Mécanique granulaire  
**Innovation**: Première étude d'humidité  
**Impact**: Applications d'ingénierie  
""")

# Quick access to saved experiments
if st.session_state.experiments:
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 💾 Expériences Sauvegardées")
    for exp_name, exp in st.session_state.experiments.items():
        meta = exp['metadata']
        st.sidebar.markdown(f"""
        **{exp_name}**
        - Eau: {meta['water_content']}%
        - Type: {meta['sphere_type']}
        - Succès: {meta['success_rate']:.1f}%
        """)
else:
    st.sidebar.markdown("---")
    st.sidebar.info("Aucune expérience sauvegardée. Utilisez la page d'analyse pour commencer.")
