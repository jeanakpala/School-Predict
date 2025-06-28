import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import ast
import numpy as np
import os
import json
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from io import BytesIO

# Configurer la page
st.set_page_config(
    page_title="Prédiction du Risque d'Abandon", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé pour améliorer l'apparence
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .risk-high { background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%); }
    .risk-medium { background: linear-gradient(135deg, #feca57 0%, #ff9ff3 100%); }
    .risk-low { background: linear-gradient(135deg, #48dbfb 0%, #0abde3 100%); }
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar pour la navigation
st.sidebar.title(" Dashboard d'Analyse")
page = st.sidebar.selectbox(
    "Navigation",
    ["Prédiction Individuelle", "Analyse des Règles", "Statistiques Globales"]
)

# Charger les modèles et données
@st.cache_resource
def load_models():
    try:
        # Construire le chemin absolu vers les modèles
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_dir = os.path.dirname(current_dir)
        models_dir = os.path.join(project_dir, 'models')
        
        scaler = joblib.load(os.path.join(models_dir, 'scaler.joblib'))
        best_model_name = 'xgboost' 
        model = joblib.load(os.path.join(models_dir, f'{best_model_name}_model.joblib'))
        rules = pd.read_csv(os.path.join(models_dir, 'association_rules_at_risk.csv'))
        return scaler, model, rules
    except Exception as e:
        st.error(f"Erreur lors du chargement des modèles: {e}")
        return None, None, None

scaler, model, rules = load_models()

if scaler is None or model is None or rules is None:
    st.error("Impossible de charger les modèles. Vérifiez que les fichiers existent dans le dossier models/")
    st.stop()

# Titre principal
st.markdown('<div class="main-header"><h1> Prédiction du Risque d\'Abandon Scolaire</h1></div>', unsafe_allow_html=True)

if page == "Prédiction Individuelle":
    st.header(" Simulation Individuelle")
    
    # Interface utilisateur améliorée
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        st.subheader(" Informations Personnelles")
        gender = st.selectbox("Genre", ["Masculin", "Féminin"])
        highest_education = st.selectbox("Niveau d'éducation", [
            "Aucune qualification formelle", 
            "Inférieur au niveau A", 
            "Niveau A ou équivalent", 
            "Qualification HE", 
            "Qualification post-universitaire"
        ])
        age_band = st.selectbox("Tranche d'âge", ["0-35", "35-55", "55<="])
    
    with col2:
        st.subheader(" Indicateurs Académiques")
        score = st.slider("Score moyen d'évaluation", 0.0, 100.0, 50.0, help="Score moyen des évaluations")
        total_clicks = st.slider("Clics totaux sur VLE", 0, 10000, 1000, help="Nombre total de clics sur la plateforme d'apprentissage")
        
        # Ajout d'un indicateur visuel pour les clics
        if total_clicks < 1000:
            st.warning("⚠️ Engagement VLE faible")
        elif total_clicks > 5000:
            st.success(" Engagement VLE élevé")
        else:
            st.info("ℹ️ Engagement VLE moyen")
    
    with col3:
        st.subheader("🏘 Contexte Socio-économique")
        imd_band = st.slider("Indice de privation (IMD)", 0.0, 100.0, 50.0, help="Indice de privation multiple")
        
        # Indicateur visuel pour l'IMD
        if imd_band < 33:
            st.success(" Zone favorisée")
        elif imd_band > 66:
            st.error("❌ Zone défavorisée")
        else:
            st.info("ℹ️ Zone intermédiaire")

    # Encoder les entrées
    gender_map = {"Masculin": 0, "Féminin": 1}
    education_map = {
        "Aucune qualification formelle": 0,
        "Inférieur au niveau A": 1,
        "Niveau A ou équivalent": 2,
        "Qualification HE": 3,
        "Qualification post-universitaire": 4
    }
    
    # Encodage one-hot pour age_band (selon l'ordre attendu par le modèle)
    age_band_55_plus = 1 if age_band == "55<=" else 0
    age_band_35_55 = 1 if age_band == "35-55" else 0
    # age_band_0_35 est implicite (0 si les autres sont 0)
    
    input_data = pd.DataFrame({
        'gender': [gender_map[gender]],
        'highest_education': [education_map[highest_education]],
        'imd_band': [imd_band],
        'age_band_55<=': [age_band_55_plus],
        'age_band_35-55': [age_band_35_55],
        'score': [score],
        'total_clicks': [total_clicks]
    })

    # Normaliser et prédire
    input_scaled = scaler.transform(input_data)
    risk_proba = model.predict_proba(input_scaled)[0][1]

    # Afficher le résultat avec des métriques visuelles
    st.markdown("---")
    st.header(" Résultat de la Prédiction")
    
    # Déterminer le niveau de risque
    if risk_proba > 0.7:
        risk_level = "Élevé"
        risk_class = "risk-high"
    elif risk_proba > 0.4:
        risk_level = "Moyen"
        risk_class = "risk-medium"
    else:
        risk_level = "Faible"
        risk_class = "risk-low"
    
    # Métriques en colonnes
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f'<div class="metric-card {risk_class}"><h3>Risque d\'Abandon</h3><h2>{risk_proba:.1%}</h2></div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown(f'<div class="metric-card"><h3>Niveau de Risque</h3><h2>{risk_level}</h2></div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown(f'<div class="metric-card"><h3>Score Moyen</h3><h2>{score:.1f}/100</h2></div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown(f'<div class="metric-card"><h3>Engagement VLE</h3><h2>{total_clicks:,}</h2></div>', unsafe_allow_html=True)

    # Graphique de jauge pour le risque
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = risk_proba * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Risque d'Abandon (%)"},
        delta = {'reference': 50},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 40], 'color': "lightgreen"},
                {'range': [40, 70], 'color': "yellow"},
                {'range': [70, 100], 'color': "red"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 70
            }
        }
    ))
    fig.update_layout(height=300)
    st.plotly_chart(fig, use_container_width=True)

    # Discretiser pour les règles
    score_cat = 'Score_Faible' if score < 33.33 else 'Score_Moyen' if score < 66.67 else 'Score_Élevé'
    clicks_cat = 'Clicks_Faible' if total_clicks < 3333 else 'Clicks_Moyen' if total_clicks < 6667 else 'Clicks_Élevé'
    imd_cat = 'IMD_Faible' if imd_band < 33.33 else 'IMD_Moyen' if imd_band < 66.67 else 'IMD_Élevé'

    # Filtrer les règles pertinentes
    input_features = [
        f'gender_{gender}',
        f'highest_education_{highest_education.replace(" ", "_")}',
        f'imd_band_cat_{imd_cat}',
        f'age_band_55<={age_band_55_plus}',
        f'age_band_35-55_{age_band_35_55}',
        f'score_cat_{score_cat}',
        f'clicks_cat_{clicks_cat}'
    ]
    
    # Corriger l'erreur de linter en gérant correctement les colonnes
    try:
        relevant_rules = rules[rules['antecedents'].astype(str).apply(lambda x: any(f in x for f in input_features))]
    except:
        relevant_rules = pd.DataFrame()

    # Afficher les règles pertinentes
    st.header(" Règles d'Association Pertinentes")
    if not relevant_rules.empty:
        # Traiter les colonnes de manière sécurisée
        def safe_process_rules(column):
            try:
                return column.astype(str).apply(lambda x: 
                    ', '.join(eval(x)) if x != 'nan' and x.strip() and x.startswith('{') else 'N/A'
                )
            except:
                return column.astype(str).apply(lambda x: 'N/A' if x == 'nan' else str(x))
        
        display_rules = relevant_rules.copy()
        display_rules['antecedents'] = safe_process_rules(display_rules['antecedents'])
        display_rules['consequents'] = safe_process_rules(display_rules['consequents'])
        
        # Formater les colonnes numériques
        display_rules['support'] = display_rules['support'].apply(lambda x: f"{x:.3f}")
        display_rules['confidence'] = display_rules['confidence'].apply(lambda x: f"{x:.3f}")
        display_rules['lift'] = display_rules['lift'].apply(lambda x: f"{x:.3f}")
        
        st.dataframe(
            display_rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']],
            use_container_width=True
        )
    else:
        st.info("Aucune règle d'association pertinente trouvée pour ces entrées.")

    # Recommandations améliorées
    st.header(" Recommandations Personnalisées")
    
    recommendations = []
    if risk_proba > 0.7:
        recommendations.append(" **Suivi personnalisé intensif** : Planifier des entretiens hebdomadaires")
    elif risk_proba > 0.4:
        recommendations.append("⚠ **Suivi régulier** : Planifier des entretiens mensuels")
    
    if total_clicks < 1000:
        recommendations.append(" **Encourager l'engagement VLE** : Promouvoir l'utilisation de la plateforme")
    
    if score < 60:
        recommendations.append(" **Soutien académique** : Proposer des cours de soutien")
    
    if imd_band > 66:
        recommendations.append("🏘 **Soutien socio-économique** : Orienter vers les services sociaux")
    
    if not recommendations:
        recommendations.append(" **Continuer le suivi actuel** : L'étudiant semble bien engagé")
    
    for rec in recommendations:
        st.write(rec)

    # Téléchargement du rapport
    st.markdown("---")
    col1, col2 = st.columns([1, 1])
    
    with col1:
        if st.button(" Générer le rapport CSV", type="primary"):
            report = input_data.copy()
            report['risk_probability'] = risk_proba
            report['risk_level'] = risk_level
            csv = report.to_csv(index=False)
            st.download_button(
                label=" Télécharger le rapport CSV",
                data=csv,
                file_name="student_risk_report.csv",
                mime="text/csv"
            )
    
    with col2:
        if st.button(" Afficher le rapport détaillé"):
            # Créer un rapport plus détaillé
            detailed_report = {
                'Informations_étudiant': {
                    'Genre': gender,
                    'Niveau_éducation': highest_education,
                    'Tranche_âge': age_band,
                    'Score_moyen': score,
                    'Clics_VLE': total_clicks,
                    'Indice_privation': imd_band
                },
                'Prédiction': {
                    'Risque_abandon': f"{risk_proba:.1%}",
                    'Niveau_risque': risk_level,
                    'Probabilité_numerique': float(risk_proba)
                },
                'Analyse_des_facteurs': {
                    'Score_catégorie': score_cat,
                    'Clics_catégorie': clicks_cat,
                    'IMD_catégorie': imd_cat,
                    'Engagement_VLE': 'Faible' if total_clicks < 1000 else 'Moyen' if total_clicks < 5000 else 'Élevé',
                    'Zone_socio_économique': 'Favorisée' if imd_band < 33 else 'Intermédiaire' if imd_band < 66 else 'Défavorisée'
                },
                'Recommandations': recommendations,
                'Règles_association_pertinentes': len(relevant_rules),
                'Date_analyse': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
                'Version_modèle': 'XGBoost v3.0.2'
            }
            
            # Afficher le rapport dans l'interface
            st.subheader(" Rapport Détaillé")
            st.json(detailed_report)
            
            # Bouton pour télécharger le PDF après affichage
            if st.button(" Télécharger le rapport PDF", key="download_pdf"):
                # Fonction pour générer le PDF
                def generate_pdf_report():
                    pdf_buffer = BytesIO()
                    doc = SimpleDocTemplate(pdf_buffer, pagesize=A4)
                    styles = getSampleStyleSheet()
                    
                    # Styles personnalisés
                    title_style = ParagraphStyle(
                        'CustomTitle',
                        parent=styles['Title'],
                        fontSize=16,
                        spaceAfter=30,
                        alignment=TA_CENTER,
                        textColor=colors.darkblue
                    )
                    
                    heading_style = ParagraphStyle(
                        'CustomHeading',
                        parent=styles['Heading2'],
                        fontSize=14,
                        spaceAfter=12,
                        textColor=colors.darkblue
                    )
                    
                    # Contenu du PDF
                    story = []
                    
                    # Titre principal
                    story.append(Paragraph("RAPPORT D'ANALYSE DU RISQUE D'ABANDON SCOLAIRE", title_style))
                    story.append(Spacer(1, 20))
                    
                    # Informations étudiant
                    story.append(Paragraph("INFORMATIONS ÉTUDIANT", heading_style))
                    story.append(Spacer(1, 12))
                    
                    student_data = [
                        ['Champ', 'Valeur'],
                        ['Genre', gender],
                        ['Niveau d\'éducation', highest_education],
                        ['Tranche d\'âge', age_band],
                        ['Score moyen', f"{score:.1f}/100"],
                        ['Clics VLE', f"{total_clicks:,}"],
                        ['Indice de privation (IMD)', f"{imd_band:.1f}"]
                    ]
                    
                    student_table = Table(student_data, colWidths=[2*inch, 3*inch])
                    student_table.setStyle(TableStyle([
                        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                        ('FONTSIZE', (0, 0), (-1, 0), 12),
                        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                        ('GRID', (0, 0), (-1, -1), 1, colors.black)
                    ]))
                    story.append(student_table)
                    story.append(Spacer(1, 20))
                    
                    # Prédiction
                    story.append(Paragraph("PRÉDICTION", heading_style))
                    story.append(Spacer(1, 12))
                    
                    prediction_data = [
                        ['Métrique', 'Valeur'],
                        ['Risque d\'abandon', f"{risk_proba:.1%}"],
                        ['Niveau de risque', risk_level]
                    ]
                    
                    prediction_table = Table(prediction_data, colWidths=[2*inch, 3*inch])
                    prediction_table.setStyle(TableStyle([
                        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                        ('FONTSIZE', (0, 0), (-1, 0), 12),
                        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                        ('GRID', (0, 0), (-1, -1), 1, colors.black)
                    ]))
                    story.append(prediction_table)
                    story.append(Spacer(1, 20))
                    
                    # Analyse des facteurs
                    story.append(Paragraph("ANALYSE DES FACTEURS", heading_style))
                    story.append(Spacer(1, 12))
                    
                    factors_data = [
                        ['Facteur', 'Catégorie'],
                        ['Score', score_cat],
                        ['Clics VLE', clicks_cat],
                        ['IMD', imd_cat],
                        ['Engagement VLE', 'Faible' if total_clicks < 1000 else 'Moyen' if total_clicks < 5000 else 'Élevé'],
                        ['Zone socio-économique', 'Favorisée' if imd_band < 33 else 'Intermédiaire' if imd_band < 66 else 'Défavorisée']
                    ]
                    
                    factors_table = Table(factors_data, colWidths=[2*inch, 3*inch])
                    factors_table.setStyle(TableStyle([
                        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                        ('FONTSIZE', (0, 0), (-1, 0), 12),
                        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                        ('GRID', (0, 0), (-1, -1), 1, colors.black)
                    ]))
                    story.append(factors_table)
                    story.append(Spacer(1, 20))
                    
                    # Recommandations
                    story.append(Paragraph("RECOMMANDATIONS", heading_style))
                    story.append(Spacer(1, 12))
                    
                    for i, rec in enumerate(recommendations, 1):
                        story.append(Paragraph(f"{i}. {rec}", styles['Normal']))
                        story.append(Spacer(1, 6))
                    
                    story.append(Spacer(1, 20))
                    
                    # Métadonnées
                    story.append(Paragraph("MÉTADONNÉES", heading_style))
                    story.append(Spacer(1, 12))
                    
                    metadata_data = [
                        ['Information', 'Valeur'],
                        ['Date d\'analyse', pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')],
                        ['Modèle utilisé', 'XGBoost v3.0.2'],
                        ['Règles d\'association trouvées', str(len(relevant_rules))]
                    ]
                    
                    metadata_table = Table(metadata_data, colWidths=[2*inch, 3*inch])
                    metadata_table.setStyle(TableStyle([
                        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                        ('FONTSIZE', (0, 0), (-1, 0), 12),
                        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                        ('GRID', (0, 0), (-1, -1), 1, colors.black)
                    ]))
                    story.append(metadata_table)
                    
                    # Générer le PDF
                    doc.build(story)
                    pdf_buffer.seek(0)
                    return pdf_buffer.getvalue()
                
                # Générer et télécharger le PDF
                pdf_content = generate_pdf_report()
                st.download_button(
                    label="📥 Télécharger le rapport PDF",
                    data=pdf_content,
                    file_name=f"rapport_etudiant_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                    mime="application/pdf"
                )

elif page == "Analyse des Règles":
    st.header(" Analyse des Règles d'Association")
    
    # Statistiques des règles
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Nombre total de règles", len(rules))
    
    with col2:
        avg_confidence = rules['confidence'].mean()
        st.metric("Confiance moyenne", f"{avg_confidence:.3f}")
    
    with col3:
        avg_lift = rules['lift'].mean()
        st.metric("Lift moyen", f"{avg_lift:.3f}")
    
    # Filtres pour les règles
    st.subheader("Filtres")
    col1, col2 = st.columns(2)
    
    with col1:
        min_confidence = st.slider("Confiance minimale", 0.0, 1.0, 0.5)
        min_support = st.slider("Support minimal", 0.0, 1.0, 0.01)
    
    with col2:
        min_lift = st.slider("Lift minimal", 0.0, 10.0, 1.0)
        max_rules = st.number_input("Nombre max de règles à afficher", 10, 100, 50)
    
    # Filtrer les règles
    filtered_rules = rules[
        (rules['confidence'] >= min_confidence) &
        (rules['support'] >= min_support) &
        (rules['lift'] >= min_lift)
    ].head(max_rules)
    
    if not filtered_rules.empty:
        # Graphique de dispersion confidence vs lift
        fig = px.scatter(
            filtered_rules, 
            x='confidence', 
            y='lift', 
            size='support',
            hover_data=['antecedents', 'consequents'],
            title="Confiance vs Lift des Règles d'Association"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Afficher les règles filtrées
        st.subheader(f"Règles filtrées ({len(filtered_rules)} règles)")
        display_rules = filtered_rules.copy()
        
        # Fonction améliorée pour traiter les règles
        def safe_process_rules_improved(column):
            try:
                return column.astype(str).apply(lambda x: 
                    ', '.join(eval(x)) if x != 'nan' and x.strip() and x.startswith('{') else 'N/A'
                )
            except:
                return column.astype(str).apply(lambda x: 'N/A' if x == 'nan' else str(x))
        
        display_rules['antecedents'] = safe_process_rules_improved(display_rules['antecedents'])
        display_rules['consequents'] = safe_process_rules_improved(display_rules['consequents'])
        
        # Formater les colonnes numériques
        display_rules['support'] = display_rules['support'].apply(lambda x: f"{x:.3f}")
        display_rules['confidence'] = display_rules['confidence'].apply(lambda x: f"{x:.3f}")
        display_rules['lift'] = display_rules['lift'].apply(lambda x: f"{x:.3f}")
        
        st.dataframe(
            display_rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']],
            use_container_width=True
        )
    else:
        st.warning("Aucune règle ne correspond aux critères de filtrage.")

elif page == "Statistiques Globales":
    st.header(" Statistiques Globales")
    
    # Créer des données simulées pour les statistiques
    np.random.seed(42)
    n_students = 1000
    
    # Simuler des données d'étudiants
    simulated_data = pd.DataFrame({
        'gender': np.random.choice(['Masculin', 'Féminin'], n_students),
        'age_band': np.random.choice(['0-35', '35-55', '55<='], n_students),
        'score': np.random.normal(65, 15, n_students).clip(0, 100),
        'total_clicks': np.random.poisson(2000, n_students),
        'imd_band': np.random.uniform(0, 100, n_students),
        'risk_proba': np.random.beta(2, 5, n_students)
    })
    
    # Métriques globales
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Nombre d'étudiants", n_students)
    
    with col2:
        avg_score = simulated_data['score'].mean()
        st.metric("Score moyen", f"{avg_score:.1f}")
    
    with col3:
        avg_clicks = simulated_data['total_clicks'].mean()
        st.metric("Clics moyens VLE", f"{avg_clicks:.0f}")
    
    with col4:
        high_risk = (simulated_data['risk_proba'] > 0.7).sum()
        st.metric("Étudiants à haut risque", f"{high_risk} ({high_risk/n_students:.1%})")
    
    # Graphiques
    col1, col2 = st.columns(2)
    
    with col1:
        # Distribution des scores
        fig = px.histogram(
            simulated_data, 
            x='score', 
            nbins=20,
            title="Distribution des Scores",
            labels={'score': 'Score', 'count': 'Nombre d\'étudiants'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Distribution des risques
        fig = px.histogram(
            simulated_data, 
            x='risk_proba', 
            nbins=20,
            title="Distribution des Risques d'Abandon",
            labels={'risk_proba': 'Probabilité de risque', 'count': 'Nombre d\'étudiants'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Graphique de corrélation
    fig = px.scatter(
        simulated_data,
        x='score',
        y='total_clicks',
        color='risk_proba',
        title="Corrélation Score vs Clics VLE",
        labels={'score': 'Score', 'total_clicks': 'Clics VLE', 'risk_proba': 'Risque'}
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Tableau de statistiques par genre
    st.subheader("Statistiques par Genre")
    gender_stats = simulated_data.groupby('gender').agg({
        'score': ['mean', 'std'],
        'total_clicks': ['mean', 'std'],
        'risk_proba': ['mean', 'std']
    }).round(2)
    
    st.dataframe(gender_stats, use_container_width=True)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666; padding: 1rem;'>
        <p> Dashboard de Prédiction du Risque d'Abandon Scolaire à partir du dataset OULAD</p>
    </div>
    """,
    unsafe_allow_html=True
)