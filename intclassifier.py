from flask import Flask, render_template, request, jsonify

from flask_cors import CORS
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack, csr_matrix
import re
import shap
app = Flask(__name__)
CORS(app)

# Load your transparency stack results
print("Loading model...")
transparency_results = joblib.load('xgboost_transparency_results.pkl')
transparency_stack = transparency_results['models']['baseline_model']
calibrated_model = transparency_results['models']['calibrated_model']
ood_detector = transparency_results['models']['ood_detector']
conformal_predictor = transparency_results['models']['conformal_predictor']

# Load the vectorizer and feature info (you'll need to save these)
# If you don't have these saved, you'll need to recreate them
try:
    feature_info = joblib.load('feature_info.pkl')
    vectorizer = feature_info['vectorizer']
    domain_feature_names = feature_info['domain_feature_names']
except:
    print("Warning: Could not load vectorizer. Using default configuration.")
    vectorizer = None
    domain_feature_names = []

def preprocess_email(email_text):
    """
    Preprocess email text and extract features matching your training pipeline
    """
    def preprocess_for_security_analysis(text):
        """Clean and prepare text to surface security-relevant patterns"""
        text = text.lower()
        
        # Extract security patterns
        security_patterns = []
        
        # Urgent phrases
        urgent_patterns = re.findall(
            r'\b(?:urgent|immediate|asap|quickly|expire|deadline|limited time|act now|expires soon)\b', 
            text
        )
        security_patterns.extend([f"urgent_{p}" for p in urgent_patterns])
        
        # Action commands
        action_patterns = re.findall(
            r'\b(?:click here|verify now|confirm|update|download|install|activate|validate|authenticate)\b', 
            text
        )
        security_patterns.extend([f"action_{p.replace(' ', '_')}" for p in action_patterns])
        
        # Authority/trust terms
        authority_patterns = re.findall(
            r'\b(?:security alert|account suspended|unauthorized access|your account|dear customer|official notice)\b', 
            text
        )
        security_patterns.extend([f"authority_{p.replace(' ', '_')}" for p in authority_patterns])
        
        # Financial lures
        financial_patterns = re.findall(
            r'\b(?:refund|payment|credit card|bank account|transaction|billing|invoice|prize|winner|money)\b', 
            text
        )
        security_patterns.extend([f"financial_{p.replace(' ', '_')}" for p in financial_patterns])
        
        # Count URLs and emails
        url_count = len(re.findall(r'http[s]?://[^\s]+', text))
        email_count = len(re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text))
        
        # Clean text
        text = re.sub(r'http[s]?://[^\s]+', ' URL_PLACEHOLDER ', text)
        text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', ' EMAIL_PLACEHOLDER ', text)
        text = re.sub(r'\b\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4}\b', ' ', text)
        text = re.sub(r'\b\d{1,2}:\d{2}(?::\d{2})?\s*(?:am|pm)?\b', ' ', text)
        text = re.sub(r'\b(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b', ' ', text)
        text = re.sub(r'\b(?:january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|jun|jul|aug|sep|oct|nov|dec)\b', ' ', text)
        text = re.sub(r'\b\d+\b(?!\s*(?:hours|days|minutes|years|months|percent|%|dollars?))', ' ', text)
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        
        # Add back security patterns
        security_text = ' '.join(security_patterns)
        final_text = f"{text} {security_text}".strip()
        
        return final_text, url_count, email_count
    
    # Preprocess text
    cleaned_text, url_count, email_count = preprocess_for_security_analysis(email_text)
    
    # Extract TF-IDF features
    if vectorizer:
        X_text = vectorizer.transform([cleaned_text])
    else:
        # Fallback: create minimal features
        X_text = csr_matrix([[len(email_text)]])
    
    # Extract domain features
    domain_features = {
        'length_log': np.log1p(len(email_text)),
        'word_count_log': np.log1p(len(email_text.split())),
        'sentence_count': len(re.split(r'[.!?]+', email_text)),
        'avg_word_length': np.mean([len(w) for w in email_text.split()] or [0]),
        'punct_ratio': sum(1 for c in email_text if not c.isalnum()) / max(len(email_text), 1),
        'digit_ratio': sum(1 for c in email_text if c.isdigit()) / max(len(email_text), 1),
        'caps_ratio': sum(1 for c in email_text if c.isupper()) / max(len(email_text), 1),
        'exclamation_count': email_text.count('!'),
        'question_count': email_text.count('?'),
        'url_count': url_count,
        'email_count': email_count,
        'dollar_count': email_text.count('$'),
        'urgent_words': len(re.findall(
            r'\b(?:urgent|immediate|asap|quickly|deadline|expire|limited|hurry)\b', 
            email_text.lower()
        )),
        'action_words': len(re.findall(
            r'\b(?:click|verify|confirm|update|download|activate|validate)\b', 
            email_text.lower()
        )),
        'fear_words': len(re.findall(
            r'\b(?:suspended|locked|unauthorized|violation|fraud|risk|danger)\b', 
            email_text.lower()
        )),
        'money_words': len(re.findall(
            r'\b(?:money|cash|refund|payment|prize|winner|reward|free)\b', 
            email_text.lower()
        )),
        'authority_claims': len(re.findall(
            r'\b(?:security team|customer service|technical support|account department)\b', 
            email_text.lower()
        )),
        'legitimacy_claims': len(re.findall(
            r'\b(?:legitimate|authorized|certified|official|verified)\b', 
            email_text.lower()
        )),
    }
    
    # Create domain feature array
    domain_array = np.array([[domain_features[name] for name in domain_feature_names]])
    
    # Combine features
    X_combined = hstack([X_text, csr_matrix(domain_array)], format='csr')
    
    return X_combined



def predict_with_transparency(X):
    # Get calibrated prediction
    calibrated_prob = calibrated_model.predict_proba(X)[0, 1]
    
    # Get OOD score
    ood_score = ood_detector.detect(X)[0]
    is_ood = ood_score > ood_detector.threshold
    
    # Get conformal band
    conformal_band = conformal_predictor.predict_bands(
        np.array([calibrated_prob]), 
        np.array([is_ood])
    )[0]
    
    # Determine classification (binary decision)
    if calibrated_prob > 0.5:
        prediction = 'phishing'
        confidence = calibrated_prob
    else:
        prediction = 'legitimate'
        confidence = 1 - calibrated_prob
    
    # Determine risk level based on ACTUAL probability
    if prediction == 'phishing':
        # Email classified as phishing
        if calibrated_prob > 0.85 or conformal_band == 'High':
            risk_level = 'high'
        elif calibrated_prob > 0.65:
            risk_level = 'moderate'
        else:
            risk_level = 'moderate'
    else:
        # Email classified as legitimate
        if calibrated_prob < 0.15 or conformal_band == 'Low':
            risk_level = 'low'
        elif calibrated_prob < 0.35:
            risk_level = 'low'
        else:
            risk_level = 'moderate'
    
    # If OOD, always treat as moderate risk (uncertain)
    if calibrated_prob > 0.95 or calibrated_prob < 0.05:
        is_ood = False 

    if is_ood:
        risk_level = 'moderate'
    
    # ========== COMPUTE SHAP VALUES FOR THIS SPECIFIC EMAIL ==========
    try:
        explainer = shap.TreeExplainer(transparency_stack)
        shap_values = explainer.shap_values(X)
        
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        
        shap_abs = np.abs(shap_values[0])
        top_indices = np.argsort(shap_abs)[-10:][::-1]
        
        # Map feature names
        feature_names = []
        if vectorizer:
            vocab = list(vectorizer.get_feature_names_out())
            feature_names.extend(vocab)
        feature_names.extend(domain_feature_names)
        
        top_features = []
        for idx in top_indices:
            if idx < len(feature_names) and shap_abs[idx] > 0.001:
                feature_name = feature_names[idx]
                shap_value = float(shap_values[0][idx])
                shap_importance = float(shap_abs[idx])
                
                if hasattr(X, 'toarray'):
                    feature_value = X.toarray()[0, idx]
                else:
                    feature_value = X[0, idx]
                
                # Create readable descriptions
                if 'urgent' in feature_name.lower():
                    readable_name = 'Urgent Language'
                    value = f'Detected: {feature_name.replace("urgent_", "")}'
                elif 'url_count' in feature_name.lower():
                    readable_name = 'URL Count'
                    value = f'{int(feature_value)} URLs'
                elif 'action' in feature_name.lower():
                    readable_name = 'Action Command'
                    value = f'{feature_name.replace("action_", "").replace("_", " ")}'
                elif 'financial' in feature_name.lower() or 'money' in feature_name.lower():
                    readable_name = 'Financial Terms'
                    value = f'{feature_name.replace("financial_", "")}'
                elif 'fear_words' in feature_name.lower():
                    readable_name = 'Fear Language'
                    value = f'{int(feature_value)} instances'
                elif 'authority' in feature_name.lower():
                    readable_name = 'Authority Claims'
                    value = f'{feature_name.replace("authority_", "").replace("_", " ")}'
                elif 'digit_ratio' in feature_name.lower():
                    readable_name = 'Digit Ratio'
                    value = f'{feature_value:.2f}'
                elif 'length_log' in feature_name.lower():
                    readable_name = 'Length Log'
                    value = f'{feature_value:.2f}'
                elif 'punct_ratio' in feature_name.lower():
                    readable_name = 'Punct Ratio'
                    value = f'{feature_value:.2f}'
                else:
                    readable_name = feature_name.replace('_', ' ').title()
                    value = f'Present (value: {feature_value:.2f})' if feature_value != 0 else 'Not present'
                
                impact_direction = 'increases' if shap_value > 0 else 'decreases'
                
                top_features.append({
                    'name': readable_name,
                    'impact': shap_importance,
                    'value': value,
                    'direction': impact_direction,
                    'shap_value': shap_value
                })
        
        top_features.sort(key=lambda x: x['impact'], reverse=True)
        
    except Exception as e:
        print(f"SHAP calculation error: {e}")
        import traceback
        traceback.print_exc()
        top_features = []
    
    # Build result - FIXED: use 'prediction' variable instead of non-existent 'calibrated_pred'
    result = {
        'prediction': prediction,  # ← FIXED
        'confidence': float(confidence),  # ← FIXED
        'riskLevel': risk_level,
        'calibrationScore': 0.945,
        'oodScore': float(ood_score),
        'conformalBand': conformal_band,
        'topFeatures': top_features[:8]
    }
    
    return result

@app.route('/')
def home():
    return render_template('client_v3.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        data = request.json
        email_text = data.get('text', '')
        
        if not email_text:
            return jsonify({'error': 'No email text provided'}), 400
        
        # Preprocess email
        X = preprocess_email(email_text)
        
        # Get predictions with transparency metrics
        result = predict_with_transparency(X)
        
        return jsonify(result)
    
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'model': 'XGBoost Transparency Stack'})

if __name__ == '__main__':
    print("Starting Email Risk Assessment API...")
    print("Model loaded successfully!")
    app.run(host='0.0.0.0', port=5002, debug=True)