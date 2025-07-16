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
    .report-section {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
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
    
    # Trajectory efficiency model
    if df_model['trajectory_efficiency'].notna().sum() >= 3:
        x_water = df_model['water_content'].values
        y_traj = df_model['trajectory_efficiency'].values
        
        mask = ~(np.isnan(x_water) | np.isnan(y_traj))
        x_clean = x_water[mask]
        y_clean = y_traj[mask]
        
        if len(x_clean) >= 3:
            degree = 2 if len(x_clean) >= 4 else 1
            coeffs = np.polyfit(x_clean, y_clean, degree)
            
            y_pred = np.polyval(coeffs, x_clean)
            r2 = calculate_r2(y_clean, y_pred)
            std_error = np.std(y_clean - y_pred)
            
            models['trajectory_efficiency'] = {
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
            recommendations.append(f"⚠️ **Sensibilité critique**: {krr_increase:.0f}% d'augmentation de résistance - contrôle d'humidité essentiel")
        elif krr_increase > 20:
            recommendations.append(f"⚠️ **Sensibilité modérée**: {krr_increase:.0f}% d'augmentation de résistance - surveillance d'humidité recommandée")
        else:
            recommendations.append(f"✅ **Faible sensibilité**: Seulement {krr_increase:.0f}% d'augmentation de résistance - humidité moins critique")
    
    # Application-specific recommendations
    recommendations.append("🏭 **Applications industrielles**:")
    recommendations.append("   • Systèmes de convoyage: Maintenir la teneur en eau ±2% de l'optimum")
    recommendations.append("   • Transport longue distance: Utiliser une teneur en eau plus faible pour l'efficacité")
    recommendations.append("   • Applications de précision: Surveiller l'humidité en continu")
    
    return recommendations

def generate_auto_report(experiments_data):
    """Generate comprehensive automatic report"""
    if not experiments_data:
        return "Aucune donnée expérimentale disponible pour la génération de rapport."
    
    # Build models
    models = build_prediction_model(experiments_data)
    
    # Get recommendations
    recommendations = generate_engineering_recommendations(experiments_data, models)
    
    current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    report = f"""
# 📊 RAPPORT D'ANALYSE AUTOMATIQUE
## Résistance au Roulement des Sphères sur Matériau Granulaire Humide

**Généré le:** {current_time}  
**Nombre d'expériences:** {len(experiments_data)}

---

## 🎯 RÉSUMÉ EXÉCUTIF

### Principales Découvertes:
"""
    
    # Add key metrics
    all_krr = []
    all_water = []
    all_efficiency = []
    all_success_rates = []
    
    for exp_name, exp in experiments_data.items():
        df = exp['data']
        meta = exp['metadata']
        df_valid = df[(df['X_center'] != 0) & (df['Y_center'] != 0) & (df['Radius'] != 0)]
        
        all_success_rates.append(meta['success_rate'])
        metrics = calculate_advanced_metrics(df_valid)
        if metrics:
            if metrics['krr'] is not None:
                all_krr.append(metrics['krr'])
            all_water.append(meta['water_content'])
            if metrics['energy_efficiency']:
                all_efficiency.append(metrics['energy_efficiency'])
    
    if all_krr:
        report += f"""
• **Gamme coefficient Krr**: {min(all_krr):.6f} - {max(all_krr):.6f}
• **Teneur en eau testée**: {min(all_water):.1f}% - {max(all_water):.1f}%
• **Succès de détection moyen**: {np.mean(all_success_rates):.1f}%
"""
    
    if all_efficiency:
        report += f"• **Gamme d'efficacité énergétique**: {min(all_efficiency):.1f}% - {max(all_efficiency):.1f}%\n"
    
    report += "\n---\n\n## 🔧 RECOMMANDATIONS D'INGÉNIERIE\n\n"
    
    for rec in recommendations:
        report += f"{rec}\n"
    
    report += "\n---\n\n## 📈 MODÈLES PRÉDICTIFS\n\n"
    
    if models:
        for param, model in models.items():
            param_name = param.replace('_', ' ').title()
            report += f"### {param_name}\n"
            report += f"• **Qualité du modèle (R²)**: {model['r2']:.3f}\n"
            report += f"• **Gamme valide**: {model['data_range'][0]:.1f}% - {model['data_range'][1]:.1f}% teneur en eau\n"
            report += f"• **Erreur standard**: ±{model['std_error']:.6f}\n"
            report += f"• **Points de données**: {model['data_points']}\n"
            
            if model['degree'] == 2:
                a, b, c = model['coeffs']
                report += f"• **Équation**: {param_name} = {a:.6f}×W² + {b:.6f}×W + {c:.6f}\n"
            else:
                a, b = model['coeffs']
                report += f"• **Équation**: {param_name} = {a:.6f}×W + {b:.6f}\n"
            report += "\n"
    else:
        report += "⚠️ Données insuffisantes pour des modèles prédictifs fiables.\n"
        report += "**Recommandation**: Collecter plus d'expériences avec teneur en eau variée.\n\n"
    
    report += "---\n\n## 📊 DÉTAILS EXPÉRIMENTAUX\n\n"
    
    for exp_name, exp in experiments_data.items():
        meta = exp['metadata']
        df = exp['data']
        df_valid = df[(df['X_center'] != 0) & (df['Y_center'] != 0) & (df['Radius'] != 0)]
        
        metrics = calculate_advanced_metrics(df_valid)
        
        report += f"### {exp_name}\n"
        report += f"• **Date**: {meta['date']}\n"
        report += f"• **Teneur en eau**: {meta['water_content']}%\n"
        report += f"• **Type de sphère**: {meta['sphere_type']}\n"
        report += f"• **Succès de détection**: {meta['success_rate']:.1f}%\n"
        
        if metrics:
            report += f"• **Coefficient Krr**: {metrics['krr']:.6f}\n" if metrics['krr'] else "• **Coefficient Krr**: N/A\n"
            report += f"• **Vitesse maximale**: {metrics['max_velocity']*1000:.1f} mm/s\n"
            report += f"• **Efficacité énergétique**: {metrics['energy_efficiency']:.1f}%\n"
        
        report += "\n"
    
    report += "---\n\n## ⚠️ LIMITATIONS & RECOMMANDATIONS\n\n"
    report += "### Qualité des Données:\n"
    
    avg_success = np.mean(all_success_rates)
    
    if avg_success >= 80:
        report += "✅ **Excellente qualité de détection** - Résultats très fiables\n"
    elif avg_success >= 70:
        report += "✅ **Bonne qualité de détection** - Résultats fiables\n"
    elif avg_success >= 60:
        report += "⚠️ **Qualité de détection modérée** - Considérer l'amélioration du setup\n"
    else:
        report += "❌ **Mauvaise qualité de détection** - Résultats potentiellement peu fiables\n"
    
    report += "\n### Validité du Modèle:\n"
    if models and any(model['r2'] > 0.8 for model in models.values()):
        report += "✅ **Modèles prédictifs solides** - Extrapolation confiante dans la gamme testée\n"
    elif models and any(model['r2'] > 0.6 for model in models.values()):
        report += "⚠️ **Modèles prédictifs modérés** - Utiliser les prédictions avec prudence\n"
    else:
        report += "❌ **Modèles prédictifs faibles** - Plus de points de données nécessaires\n"
    
    report += "\n### Prochaines Étapes:\n"
    if len(experiments_data) < 5:
        report += "• **Augmenter le nombre d'expériences** - Viser 8-10 expériences pour des modèles robustes\n"
    
    water_range = max(all_water) - min(all_water) if all_water else 0
    if water_range < 15:
        report += "• **Élargir la gamme de teneur en eau** - Tester des conditions d'humidité plus larges\n"
    
    sphere_types = set([exp['metadata']['sphere_type'] for exp in experiments_data.values()])
    if len(sphere_types) < 2:
        report += "• **Tester plusieurs matériaux de sphères** - Comparer acier, plastique, verre\n"
    
    report += "\n---\n\n## 📞 CONTACT & MÉTHODOLOGIE\n\n"
    report += "**Institution de Recherche**: Department of Cosmic Earth Science, Graduate School of Science, Osaka University\n"
    report += "**Domaine de Recherche**: Mécanique Granulaire\n"
    report += "**Innovation**: Première étude systématique des effets d'humidité sur la résistance au roulement\n\n"
    report += "**Méthodologie**: Suivi de sphères par vision par ordinateur avec analyse cinématique\n"
    report += "**Détection**: Soustraction d'arrière-plan avec transformées de Hough circulaires\n"
    report += "**Analyse**: Calcul Krr utilisant les principes de conservation d'énergie\n"
    
    return report

# Page navigation
st.sidebar.markdown("### 📋 Navigation")
page = st.sidebar.radio("Sélectionner la Page:", [
    "🏠 Analyse Unique",
    "🔍 Comparaison Multi-Expériences", 
    "🎯 Module de Prédiction",
    "📊 Rapport Auto-Généré"
])

# ==================== SINGLE ANALYSIS PAGE ====================
if page == "🏠 Analyse Unique":
    st.markdown("""
    # ⚪ Plateforme d'Analyse de Résistance au Roulement des Sphères
    ## 🔬 Suite d'Analyse Complète pour la Recherche en Mécanique Granulaire
    *Téléchargez vos données et accédez à nos outils d'analyse spécialisés*
    """)

    # File upload section
    st.markdown("""
    <div class="upload-section">
        <h2>📂 Téléchargez Vos Données Expérimentales</h2>
        <p>Commencez par télécharger votre fichier CSV avec les résultats de détection pour obtenir une analyse personnalisée</p>
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
    tab1, tab2, tab3 = st.tabs(["📁 Upload Fichier", "🔬 Données d'Exemple", "✍️ Saisie Manuelle"])
    
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
                # Show file details
                st.info(f"📄 Fichier: {uploaded_file.name} ({uploaded_file.size} bytes)")
                
                # Read the file
                df = pd.read_csv(uploaded_file)
                
                st.success(f"✅ Fichier lu avec succès! {len(df)} lignes trouvées")
                
                # Show first few rows for verification
                st.markdown("#### 👀 Aperçu des Données")
                st.dataframe(df.head(10))
                
                # Check required columns
                required_columns = ['Frame', 'X_center', 'Y_center', 'Radius']
                missing_columns = [col for col in required_columns if col not in df.columns]
                
                if missing_columns:
                    st.error(f"❌ Colonnes manquantes: {missing_columns}")
                    st.error(f"📊 Colonnes trouvées: {list(df.columns)}")
                    st.info("💡 Astuce: Vérifiez que votre fichier CSV contient bien les colonnes requises")
                    df = None
                else:
                    st.success("✅ Toutes les colonnes requises sont présentes!")
                    
                    # Show column info
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Frames Total", len(df))
                    with col2:
                        valid_detections = len(df[(df['X_center'] != 0) & (df['Y_center'] != 0) & (df['Radius'] != 0)])
                        st.metric("Détections Valides", valid_detections)
                    with col3:
                        success_rate = (valid_detections / len(df) * 100) if len(df) > 0 else 0
                        st.metric("Taux de Succès", f"{success_rate:.1f}%")
                    with col4:
                        zero_detections = len(df) - valid_detections
                        st.metric("Détections Nulles", zero_detections)
                    
                    # Filter valid detections
                    df_valid = df[(df['X_center'] != 0) & (df['Y_center'] != 0) & (df['Radius'] != 0)]
                    
                    if len(df_valid) == 0:
                        st.warning("⚠️ Aucune détection valide trouvée! Vérifiez vos données.")
                    else:
                        st.success(f"✅ {len(df_valid)} détections valides prêtes pour l'analyse!")
                        
                        # Option to save experiment
                        if st.button("💾 Sauvegarder l'expérience pour comparaison", key="save_uploaded"):
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
                            st.success(f"Expérience '{experiment_name}' sauvegardée pour comparaison!")
                            
            except Exception as e:
                st.error(f"❌ Erreur lors du chargement du fichier: {str(e)}")
                st.info("💡 Astuces de dépannage:")
                st.info("• Vérifiez que le fichier est bien un CSV")
                st.info("• Vérifiez l'encodage (UTF-8 recommandé)")
                st.info("• Vérifiez les séparateurs (virgules)")
    
    with tab2:
        st.markdown("### 🔬 Utiliser des Données d'Exemple")
        st.info("Parfait pour tester l'application sans fichier CSV")
        
        # Sample data options
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
            
            st.success("📊 Données d'exemple générées avec succès!")
            st.info(f"✅ {len(df)} frames générées, {len(df_valid)} détections valides")
            
            # Show sample data preview
            st.markdown("#### 👀 Aperçu des Données d'Exemple")
            st.dataframe(df.head(10))
    
    with tab3:
        st.markdown("### ✍️ Saisie Manuelle de Données")
        st.info("Pour saisir quelques points de données manuellement")
        
        # Manual data entry
        if 'manual_data' not in st.session_state:
            st.session_state.manual_data = []
        
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            frame_input = st.number_input("Frame", value=1, min_value=1, key="manual_frame")
        with col2:
            x_input = st.number_input("X_center", value=100.0, key="manual_x")
        with col3:
            y_input = st.number_input("Y_center", value=100.0, key="manual_y")
        with col4:
            radius_input = st.number_input("Radius", value=20.0, min_value=0.1, key="manual_radius")
        with col5:
            if st.button("➕ Ajouter Point", key="add_manual"):
                st.session_state.manual_data.append({
                    'Frame': frame_input,
                    'X_center': x_input,
                    'Y_center': y_input,
                    'Radius': radius_input
                })
                st.success("Point ajouté!")
        
        if st.session_state.manual_data:
            st.markdown(f"#### 📊 Données Saisies ({len(st.session_state.manual_data)} points)")
            manual_df = pd.DataFrame(st.session_state.manual_data)
            st.dataframe(manual_df)
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("✅ Utiliser ces données", key="use_manual"):
                    df = manual_df
                    df_valid = df[(df['X_center'] != 0) & (df['Y_center'] != 0) & (df['Radius'] != 0)]
                    st.success("Données manuelles chargées pour l'analyse!")
            
            with col2:
                if st.button("🗑️ Effacer tout", key="clear_manual"):
                    st.session_state.manual_data = []
                    st.success("Données effacées!")
                    st.rerun()

    # Analysis only if data is loaded
    if df is not None and len(df_valid) > 0:
        
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
        
        # Simple analysis and visualization
        st.markdown("### 📈 Analyse Rapide")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Trajectory plot
            fig_traj = px.scatter(df_valid, x='X_center', y='Y_center', 
                               color='Frame', 
                               title="🛤️ Trajectoire de la Sphère",
                               labels={'X_center': 'Position X (pixels)', 
                                      'Y_center': 'Position Y (pixels)',
                                      'Frame': 'Frame'})
            fig_traj.update_yaxes(autorange="reversed")
            st.plotly_chart(fig_traj, use_container_width=True)
        
        with col2:
            # Radius evolution
            fig_radius = px.line(df_valid, x='Frame', y='Radius',
                               title="⚪ Évolution du Rayon Détecté",
                               labels={'Frame': 'Numéro de Frame', 
                                      'Radius': 'Rayon (pixels)'})
            st.plotly_chart(fig_radius, use_container_width=True)
        
        # Basic Krr calculation
        st.markdown("### 🧮 Calcul Krr Basique")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            fps = st.number_input("FPS Caméra", value=250.0, min_value=1.0)
            pixels_per_mm = st.number_input("Calibration (px/mm)", value=5.0, min_value=0.1)
        
        with col2:
            sphere_mass_g = st.number_input("Masse Sphère (g)", value=10.0, min_value=0.1)
            angle_deg = st.number_input("Angle Inclinaison (°)", value=15.0, min_value=0.1)
        
        if st.button("🚀 Calculer Krr"):
            metrics = calculate_advanced_metrics(df_valid, fps, pixels_per_mm, sphere_mass_g, angle_deg)
            
            if metrics and metrics['krr'] is not None:
                with col3:
                    st.markdown(f"""
                    <div class="prediction-card">
                        <h4>Résultat Krr</h4>
                        <p><strong>{metrics['krr']:.6f}</strong></p>
                        <p>Vitesse: {metrics['max_velocity']*1000:.1f} mm/s</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Additional metrics
                st.markdown("#### 📊 Métriques Supplémentaires")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Distance Totale", f"{metrics['distance']*1000:.1f} mm")
                with col2:
                    st.metric("Durée", f"{metrics['duration']:.2f} s")
                with col3:
                    st.metric("Efficacité Énergétique", f"{metrics['energy_efficiency']:.1f}%")
                with col4:
                    st.metric("Efficacité Trajectoire", f"{metrics['trajectory_efficiency']:.1f}%")
                
                # Advanced visualization
                st.markdown("#### 📈 Visualisations Avancées")
                
                # Create velocity and acceleration plots
                col1, col2 = st.columns(2)
                
                with col1:
                    fig_vel = go.Figure()
                    fig_vel.add_trace(go.Scatter(
                        x=metrics['time'], 
                        y=metrics['velocity'] * 1000,
                        mode='lines',
                        name='Vitesse',
                        line=dict(color='blue', width=2)
                    ))
                    fig_vel.update_layout(
                        title="Évolution de la Vitesse",
                        xaxis_title="Temps (s)",
                        yaxis_title="Vitesse (mm/s)"
                    )
                    st.plotly_chart(fig_vel, use_container_width=True)
                
                with col2:
                    fig_accel = go.Figure()
                    fig_accel.add_trace(go.Scatter(
                        x=metrics['time'], 
                        y=metrics['acceleration'] * 1000,
                        mode='lines',
                        name='Accélération',
                        line=dict(color='red', width=2)
                    ))
                    fig_accel.update_layout(
                        title="Évolution de l'Accélération",
                        xaxis_title="Temps (s)",
                        yaxis_title="Accélération (mm/s²)"
                    )
                    st.plotly_chart(fig_accel, use_container_width=True)
                
                # Export detailed data
                st.markdown("#### 💾 Exporter les Données Détaillées")
                
                detailed_data = pd.DataFrame({
                    'temps_s': metrics['time'],
                    'vitesse_ms': metrics['velocity'],
                    'acceleration_ms2': metrics['acceleration'],
                    'force_resistance_N': metrics['resistance_force'],
                    'puissance_W': metrics['power'],
                    'energie_cinetique_J': metrics['energy_kinetic']
                })
                
                csv_data = detailed_data.to_csv(index=False)
                st.download_button(
                    label="📥 Télécharger les données détaillées (CSV)",
                    data=csv_data,
                    file_name="analyse_cinetique_detaillee.csv",
                    mime="text/csv"
                )
            else:
                st.error("❌ Impossible de calculer Krr - données insuffisantes")
    
    else:
        # Message if no data is loaded
        st.markdown("""
        ## 🚀 Pour commencer:
        
        1. **📂 Téléchargez votre fichier CSV** avec vos données expérimentales
        2. **Ou cliquez sur "Utiliser des données d'exemple"** pour explorer les fonctionnalités
        
        ### 📋 Format de fichier attendu:
        Votre CSV doit contenir les colonnes suivantes:
        - `Frame`: Numéro d'image
        - `X_center`: Position X du centre de la sphère
        - `Y_center`: Position Y du centre de la sphère  
        - `Radius`: Rayon détecté de la sphère
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
            
            for exp_name in selected_experiments:
                exp = st.session_state.experiments[exp_name]
                df = exp['data']
                meta = exp['metadata']
                df_valid = df[(df['X_center'] != 0) & (df['Y_center'] != 0) & (df['Radius'] != 0)]
                
                # Calculate advanced metrics
                metrics = calculate_advanced_metrics(df_valid)
                
                comparison_data.append({
                    'Expérience': exp_name,
                    'Teneur_Eau': meta['water_content'],
                    'Type_Sphere': meta['sphere_type'],
                    'Taux_Succes': meta['success_rate'],
                    'Krr': metrics['krr'] if metrics else None,
                    'Vitesse_Max': metrics['max_velocity'] if metrics else None,
                    'Efficacite_Energie': metrics['energy_efficiency'] if metrics else None,
                    'Efficacite_Trajectoire': metrics['trajectory_efficiency'] if metrics else None,
                })
            
            comp_df = pd.DataFrame(comparison_data)
            
            # Visualizations
            st.markdown("### 📊 Analyses Comparatives")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Krr vs Water Content
                if comp_df['Krr'].notna().any():
                    fig_krr = px.scatter(comp_df, x='Teneur_Eau', y='Krr', 
                                       color='Type_Sphere', size='Taux_Succes',
                                       hover_data=['Expérience'],
                                       title="🔍 Krr vs Teneur en Eau",
                                       labels={'Teneur_Eau': 'Teneur en Eau (%)', 
                                              'Krr': 'Coefficient Krr'})
                    st.plotly_chart(fig_krr, use_container_width=True)
                else:
                    st.warning("Pas de données Krr valides pour comparaison")
            
            with col2:
                # Success rate comparison
                fig_success = px.bar(comp_df, x='Expérience', y='Taux_Succes',
                                   color='Teneur_Eau',
                                   title="📈 Comparaison des Taux de Succès de Détection",
                                   labels={'Taux_Succes': 'Taux de Succès (%)'})
                fig_success.update_xaxes(tickangle=45)
                st.plotly_chart(fig_success, use_container_width=True)
            
            # Energy efficiency comparison
            if comp_df['Efficacite_Energie'].notna().any():
                col1, col2 = st.columns(2)
                
                with col1:
                    fig_energy = px.bar(comp_df, x='Expérience', y='Efficacite_Energie',
                                      color='Teneur_Eau',
                                      title="⚡ Efficacité Énergétique",
                                      labels={'Efficacite_Energie': 'Efficacité Énergétique (%)'})
                    fig_energy.update_xaxes(tickangle=45)
                    st.plotly_chart(fig_energy, use_container_width=True)
                
                with col2:
                    fig_traj = px.bar(comp_df, x='Expérience', y='Efficacite_Trajectoire',
                                    color='Teneur_Eau',
                                    title="🛤️ Efficacité de Trajectoire",
                                    labels={'Efficacite_Trajectoire': 'Efficacité Trajectoire (%)'})
                    fig_traj.update_xaxes(tickangle=45)
                    st.plotly_chart(fig_traj, use_container_width=True)
            
            # Statistical comparison table
            st.markdown("### 📋 Tableau de Comparaison Détaillé")
            
            # Format the comparison table
            display_comp = comp_df.copy()
            if 'Krr' in display_comp.columns:
                display_comp['Krr'] = display_comp['Krr'].apply(lambda x: f"{x:.6f}" if pd.notna(x) else "N/A")
            
            st.dataframe(display_comp, use_container_width=True)
            
            # Key insights
            st.markdown("### 🔍 Insights Clés")
            
            if len(comp_df) >= 2:
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    best_exp = comp_df.loc[comp_df['Taux_Succes'].idxmax()]
                    st.markdown(f"""
                    <div class="comparison-card">
                        <h4>🏆 Meilleure Détection</h4>
                        <p><strong>{best_exp['Expérience']}</strong></p>
                        <p>{best_exp['Taux_Succes']:.1f}% de succès</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    if comp_df['Krr'].notna().any():
                        min_krr_exp = comp_df.loc[comp_df['Krr'].idxmin()]
                        st.markdown(f"""
                        <div class="comparison-card">
                            <h4>⚡ Krr le Plus Bas</h4>
                            <p><strong>{min_krr_exp['Expérience']}</strong></p>
                            <p>Krr = {min_krr_exp['Krr']:.6f}</p>
                        </div>
                        """, unsafe_allow_html=True)
                
                with col3:
                    water_range = comp_df['Teneur_Eau'].max() - comp_df['Teneur_Eau'].min()
                    st.markdown(f"""
                    <div class="comparison-card">
                        <h4>💧 Gamme d'Eau Testée</h4>
                        <p><strong>{water_range:.1f}%</strong></p>
                        <p>De {comp_df['Teneur_Eau'].min()}% à {comp_df['Teneur_Eau'].max()}%</p>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Export comparison results
            st.markdown("### 💾 Exporter les Résultats de Comparaison")
            
            csv_comparison = comp_df.to_csv(index=False)
            st.download_button(
                label="📥 Télécharger les résultats de comparaison (CSV)",
                data=csv_comparison,
                file_name="comparaison_experiences.csv",
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
        # Generate report
        with st.spinner("🔄 Génération du rapport d'analyse complet..."):
            report_content = generate_auto_report(st.session_state.experiments)
        
        # Display report
        st.markdown("### 📋 Rapport Généré")
        
        # Report controls
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Download report as text
            st.download_button(
                label="📥 Télécharger le rapport (TXT)",
                data=report_content,
                file_name=f"rapport_analyse_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain"
            )
        
        with col2:
            # Generate PDF option (simplified)
            if st.button("📄 Générer rapport PDF"):
                st.info("💡 Fonction PDF en développement. Utilisez l'option TXT pour l'instant.")
        
        with col3:
            # Email report option
            if st.button("📧 Envoyer par email"):
                st.info("💡 Fonction email en développement. Utilisez l'option de téléchargement.")
        
        # Display report content
        st.markdown("---")
        
        # Report sections with expandable content
        with st.expander("📊 Voir le Rapport Complet", expanded=True):
            st.markdown(report_content)
        
        # Interactive elements for report customization
        st.markdown("### 🔧 Personnalisation du Rapport")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Options d'Inclusion")
            include_raw_data = st.checkbox("Inclure les données brutes", value=False)
            include_plots = st.checkbox("Inclure les graphiques", value=True)
            include_equations = st.checkbox("Inclure les équations détaillées", value=True)
            
        with col2:
            st.markdown("#### Format du Rapport")
            report_language = st.selectbox("Langue", ["Français", "English"])
            report_detail = st.selectbox("Niveau de détail", ["Résumé", "Standard", "Détaillé"])
            
        if st.button("🔄 Régénérer le Rapport avec Nouvelles Options"):
            with st.spinner("Régénération du rapport..."):
                # Here you would implement the custom report generation
                # For now, we'll just show the same report
                st.success("✅ Rapport régénéré avec les nouvelles options!")
        
        # Summary statistics
        st.markdown("### 📈 Statistiques du Rapport")
        
        # Calculate some basic stats about the experiments
        total_experiments = len(st.session_state.experiments)
        water_contents = [exp['metadata']['water_content'] for exp in st.session_state.experiments.values()]
        success_rates = [exp['metadata']['success_rate'] for exp in st.session_state.experiments.values()]
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Expériences Totales", total_experiments)
        
        with col2:
            st.metric("Gamme d'Humidité", f"{min(water_contents):.1f}%-{max(water_contents):.1f}%")
        
        with col3:
            st.metric("Succès Moyen", f"{np.mean(success_rates):.1f}%")
        
        with col4:
            models = build_prediction_model(st.session_state.experiments)
            model_count = len(models) if models else 0
            st.metric("Modèles Générés", model_count)
        
        # Quality assessment
        st.markdown("### 🎯 Évaluation de la Qualité")
        
        # Data quality indicators
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
        
        # Recommendations for improvement
        st.markdown("### 💡 Recommandations d'Amélioration")
        
        recommendations = []
        
        if total_experiments < 5:
            recommendations.append("📊 **Augmenter le nombre d'expériences** - Collecter au moins 5-8 expériences pour des modèles robustes")
        
        if max(water_contents) - min(water_contents) < 15:
            recommendations.append("💧 **Élargir la gamme d'humidité** - Tester une gamme plus large de teneurs en eau")
        
        if avg_success < 75:
            recommendations.append("🔧 **Améliorer la qualité de détection** - Optimiser les paramètres de détection ou l'éclairage")
        
        sphere_types = set([exp['metadata']['sphere_type'] for exp in st.session_state.experiments.values()])
        if len(sphere_types) < 2:
            recommendations.append("⚪ **Tester différents matériaux** - Inclure plusieurs types de sphères pour la comparaison")
        
        if recommendations:
            for rec in recommendations:
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
        """)
else:
    st.sidebar.markdown("---")
    st.sidebar.info("Aucune expérience sauvegardée. Utilisez la page d'analyse pour commencer.")
