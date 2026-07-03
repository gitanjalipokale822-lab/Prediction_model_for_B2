import streamlit as st
import pickle
import numpy as np
import pandas as pd
import time

# Set up page configurations with a wide layout and custom icon
st.set_page_config(
    page_title="Student Performance Classifier",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- INJECT BEAUTIFUL CSS CUSTOM STYLING & ANIMATIONS ---
st.markdown("""
<style>
    /* Gradient Background Effect */
    .stApp {
        background: linear-gradient(135deg, #121214 0%, #1a1a24 100%);
        color: #f0f2f6;
    }
    
    /* Elegant Title Styling with Glow Animation */
    @keyframes glow {
        0% { text-shadow: 0 0 10px rgba(78, 115, 223, 0.5); }
        50% { text-shadow: 0 0 20px rgba(78, 115, 223, 0.8), 0 0 30px rgba(28, 200, 138, 0.6); }
        100% { text-shadow: 0 0 10px rgba(78, 115, 223, 0.5); }
    }
    .main-title {
        font-family: 'Helvetica Neue', Arial, sans-serif;
        font-size: 3rem !important;
        font-weight: 800;
        background: linear-gradient(45deg, #4e73df, #1cc88a);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 5px;
        animation: glow 4s infinite ease-in-out;
    }
    
    /* Sliding/Fading Animation for Cards */
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    /* Custom Input Section Card Style */
    .metric-card {
        background-color: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 25px;
        border-radius: 16px;
        margin-bottom: 20px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.2);
        backdrop-filter: blur(4px);
        animation: fadeInUp 0.6s ease-out;
        transition: transform 0.3s ease, border-color 0.3s ease;
    }
    .metric-card:hover {
        transform: translateY(-5px);
        border-color: rgba(78, 115, 223, 0.5);
    }
    
    /* Section Headings Styling */
    .section-header {
        font-size: 1.4rem;
        font-weight: 600;
        color: #4e73df;
        margin-bottom: 15px;
        display: flex;
        align-items: center;
        gap: 10px;
    }

    /* Pulse animation for interactive buttons */
    .stButton>button {
        width: 100%;
        background: linear-gradient(45deg, #4e73df, #224abe) !important;
        color: white !important;
        border: none !important;
        padding: 12px 24px !important;
        font-size: 1.1rem !important;
        font-weight: bold !important;
        border-radius: 10px !important;
        transition: all 0.3s ease-in-out !important;
        box-shadow: 0 4px 15px rgba(78, 115, 223, 0.4) !important;
    }
    .stButton>button:hover {
        transform: scale(1.02);
        box-shadow: 0 6px 20px rgba(78, 115, 223, 0.6) !important;
        background: linear-gradient(45deg, #224abe, #1cc88a) !important;
    }
</style>
""", unsafe_allow_html=True)

# --- LOAD TRAINED ML MODEL ---
@st.cache_resource
def load_ml_model():
    try:
        with open("model (9).pkl", "rb") as file:
            model = pickle.load(file)
        return model
    except FileNotFoundError:
        st.error("❌ 'model (9).pkl' file was not found. Please verify it is in the active app script directory.")
        return None

classifier_model = load_ml_model()

# Header Intro Banner
st.markdown('<p class="main-title">🎓 Student Performance Analytics Portal</p>', unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #a0aec0; margin-bottom: 40px;'>Predict and analyze category targets interactively using advanced Gradient Boosting classification.</p>", unsafe_allow_html=True)

if classifier_model is not None:
    # Form initialization to structured multi-column sections
    with st.form("prediction_form"):
        
        # Split inputs cleanly into 3 distinct UI visual sections
        col1, col2, col3 = st.columns(3, gap="large")
        
        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown('<p class="section-header">👤 Profile Features</p>', unsafe_allow_html=True)
            gender = st.selectbox("Biological Gender", options=["Male", "Female"])
            age = st.slider("Age of Student", min_value=10, max_value=30, value=17, step=1)
            parent_education = st.selectbox(
                "Parental Education Level",
                options=["High School", "Associate's Degree", "Bachelor's Degree", "Master's Degree", "PhD"]
            )
            st.markdown('</div>', unsafe_allow_html=True)
            
        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown('<p class="section-header">📚 Engagement & Tech</p>', unsafe_allow_html=True)
            study_hours_per_week = st.slider("Weekly Study Hours", min_value=0, max_value=60, value=15, step=1)
            attendance_rate = st.slider("Attendance Rate (%)", min_value=0, max_value=100, value=88, step=1)
            internet_access = st.radio("Has Home Internet?", options=["Yes", "No"], horizontal=True)
            extracurricular = st.radio("Participates in Extracurriculars?", options=["Yes", "No"], horizontal=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
        with col3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown('<p class="section-header">📊 Academic Scores</p>', unsafe_allow_html=True)
            previous_score = st.number_input("Previous Term Exam Score", min_value=0.0, max_value=100.0, value=75.0, step=0.5)
            final_score = st.number_input("Current Final Exam Score Output", min_value=0.0, max_value=100.0, value=78.0, step=0.5)
            st.markdown('</div>', unsafe_allow_html=True)
            
        # Submit execution trigger
        submit_btn = st.form_submit_button(label="🔮 RUN PREDICTION TARGET")

    if submit_btn:
        # --- FEATURE PREPROCESSING DATA CONVERSIONS ---
        # Map categorical configurations systematically back into exact matching structures the binary trees expect
        gender_encoded = 1 if gender == "Male" else 0
        internet_encoded = 1 if internet_access == "Yes" else 0
        extracurricular_encoded = 1 if extracurricular == "Yes" else 0
        
        # Mapping for parental instruction categories ordinal hierarchies
        edu_mapping = {"High School": 1, "Associate's Degree": 2, "Bachelor's Degree": 3, "Master's Degree": 4, "PhD": 5}
        parent_edu_encoded = edu_mapping[parent_education]
        
        # Build DataFrame with proper model feature naming labels matching model structure metadata
        feature_vector = pd.DataFrame([{
            'gender': gender_encoded,
            'age': age,
            'study_hours_per_week': study_hours_per_week,
            'attendance_rate': attendance_rate,
            'parent_education': parent_edu_encoded,
            'internet_access': internet_encoded,
            'extracurricular': extracurricular_encoded,
            'previous_score': previous_score,
            'final_score': final_score
        }])
        
        # --- EXECUTE PREDICTION AND ANIMATIONS ---
        with st.spinner("Processing architectural features through model decision steps..."):
            time.sleep(1.2)  # Short artificial delay to make the transition smoother
            
            # Predict labels & execution metrics probabilities
            predicted_class = classifier_model.predict(feature_vector)[0]
            
            try:
                predicted_probabilities = classifier_model.predict_proba(feature_vector)[0]
                has_probabilities = True
            except AttributeError:
                has_probabilities = False
                
        # --- OUTPUT PREDICTION RESULTS PANEL ---
        st.markdown("<h3 style='text-align: center; margin-top: 30px; color:#1cc88a;'>🎯 Classification Analysis Result</h3>", unsafe_allow_html=True)
        
        # Success effects
        try:
            from streamlit_confetti import confetti
            confetti()
        except ImportError:
            st.balloons()
            
        # UI Dynamic Card Layout for displaying classification tags
        res_col1, res_col2 = st.columns([1, 2])
        with res_col1:
            st.markdown(f"""
            <div style="background-color: #1e1e2f; padding: 30px; border-radius: 12px; border-left: 5px solid #1cc88a; text-align: center; box-shadow: 0 4px 15px rgba(0,0,0,0.3);">
                <span style="font-size: 1.1rem; text-transform: uppercase; letter-spacing: 1px; color: #a0aec0;">Predicted Category</span>
                <h1 style="font-size: 3.5rem; color: #ffffff; margin: 10px 0;">{predicted_class}</h1>
            </div>
            """, unsafe_allow_html=True)
            
        with res_col2:
            if has_probabilities:
                st.markdown("<p style='margin-bottom: 5px; font-weight: 600; color: #f0f2f6;'>Model Confidence Distributions:</p>", unsafe_allow_html=True)
                for index, class_label in enumerate(classifier_model.classes_):
                    probability_val = predicted_probabilities[index]
                    # Format dynamic bar meters
                    st.markdown(f"**Class Label {class_label}**: `{probability_val * 100:.2f}%` confidence")
                    st.progress(float(probability_val))
            else:
                st.info("ℹ️ Model structure handles absolute deterministic splits without explicit probability estimation arrays.")
