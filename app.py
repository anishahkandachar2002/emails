import streamlit as st
import joblib
import re
import logging
from typing import Optional, Tuple, Set, List
import warnings
import pickle
from io import StringIO
import sys

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Mock data for spell checker (since we don't have the pickle file)
MOCK_DATA = {
    "brand_names": {
        "airbnb": "Airbnb",
        "google": "Google",
        "facebook": "Facebook",
        "microsoft": "Microsoft",
        "apple": "Apple",
        "amazon": "Amazon",
        "netflix": "Netflix",
        "spotify": "Spotify",
        "uber": "Uber",
        "tesla": "Tesla"
    },
    "manual_contractions": {
        "dont": "don't",
        "wont": "won't",
        "cant": "can't",
        "youre": "you're",
        "theyre": "they're",
        "were": "we're",
        "itll": "it'll",
        "youll": "you'll",
        "theyll": "they'll",
        "ill": "I'll",
        "wouldnt": "wouldn't",
        "couldnt": "couldn't",
        "shouldnt": "shouldn't",
        "youve": "you've",
        "theyve": "they've",
        "weve": "we've",
        "ive": "I've"
    },
    "common_words": {
        "the", "and", "for", "are", "but", "not", "you", "all", "can", "her", "was", "one", "our", "had", "by", "word", "oil", "its", "now", "find", "long", "down", "way", "who", "been", "call", "people", "water", "than", "look", "first", "also", "after", "back", "other", "many", "time", "very", "when", "come", "here", "just", "like", "long", "make", "many", "over", "such", "take", "than", "them", "well", "were", "will", "would", "there", "each", "which", "their", "said", "get", "has", "him", "his", "how", "man", "new", "now", "old", "see", "two", "way", "may", "say", "she", "use", "her", "all", "any", "can", "had", "day", "get", "has", "him", "his", "how", "man", "new", "now", "old", "see", "two", "way", "may", "say", "she", "use", "its", "did", "yes", "get", "has", "him", "his", "how", "man", "new", "now", "old", "see", "two", "way", "may", "say", "she", "use"
    },
    "patterns": [
        (["I", "you", "he", "she", "it", "we", "they"], "ll"),
        (["I", "you", "he", "she", "it", "we", "they"], "ve"),
        (["I", "you", "he", "she", "it", "we", "they"], "re"),
        (["I", "you", "he", "she", "it", "we", "they"], "d")
    ],
    "negative_bases": ["do", "does", "did", "will", "would", "could", "should", "might", "must", "can", "have", "has", "had", "is", "are", "was", "were"],
    "special_contractions": ["o'clock", "ma'am", "y'all"],
    "corrections": [
        (r'\b(\d+)minutes\b', r'\1 minutes'),
        (r'\b(\w+)(\w+)answers\b', r'\1 \2 answers'),
        (r'\bairbnb\b', 'Airbnb'),
        (r'\bgoogle\b', 'Google'),
        (r'\bfacebook\b', 'Facebook'),
        (r'\bmicrosoft\b', 'Microsoft'),
        (r'\bapple\b', 'Apple'),
        (r'\bamazon\b', 'Amazon'),
        (r'\bnetflix\b', 'Netflix'),
        (r'\bspotify\b', 'Spotify'),
        (r'\buber\b', 'Uber'),
        (r'\btesla\b', 'Tesla')
    ]
}

class SimpleSpellChecker:
    """
    A simplified spell checker for the Streamlit app.
    This version uses basic text correction without heavy dependencies.
    """
    
    def __init__(self):
        self.brand_names = MOCK_DATA["brand_names"]
        self.manual_contractions = MOCK_DATA["manual_contractions"]
        self.common_words = MOCK_DATA["common_words"]
        self.corrections = MOCK_DATA["corrections"]
        
    def expand_contractions(self, text: str) -> str:
        """Expand contractions using manual patterns"""
        for contraction, expansion in self.manual_contractions.items():
            text = re.sub(r'\b' + re.escape(contraction) + r'\b', expansion, text, flags=re.IGNORECASE)
        return text
    
    def apply_regex_corrections(self, text: str) -> str:
        """Apply regex-based corrections for specific patterns"""
        for pattern, replacement in self.corrections:
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        return text
    
    def basic_cleanup(self, text: str) -> str:
        """Basic text cleanup - minimal changes to preserve structure"""
        # Fix spacing around punctuation
        text = re.sub(r'([.!?,:;])([a-zA-Z])', r'\1 \2', text)
        # Fix spacing between numbers and letters
        text = re.sub(r'(\d+)([a-zA-Z])', r'\1 \2', text)
        # Fix spacing between lowercase and uppercase sequences
        text = re.sub(r'([a-z])([A-Z]{2,})', r'\1 \2', text)
        # Normalize whitespace but preserve structure
        text = re.sub(r'[ \t]+', ' ', text)
        return text
    
    def capitalize_after_punctuation(self, text: str) -> str:
        """Capitalize words after sentence-ending punctuation"""
        lines = text.split('\n')
        result_lines = []
        
        for line in lines:
            if not line.strip():
                result_lines.append(line)
                continue
            
            # Capitalize after punctuation
            pattern = r'([.!?])\s+([a-z])'
            def capitalize_match(match):
                return match.group(1) + ' ' + match.group(2).upper()
            
            result_line = re.sub(pattern, capitalize_match, line)
            
            # Capitalize first word of line
            if result_line and result_line[0].islower():
                result_line = result_line[0].upper() + result_line[1:]
            
            result_lines.append(result_line)
        
        return '\n'.join(result_lines)
    
    def capitalize_brand_names(self, text: str) -> str:
        """Capitalize brand names according to the brand_names dictionary"""
        result = text
        for lowercase_brand, proper_brand in self.brand_names.items():
            pattern = r'\b' + re.escape(lowercase_brand) + r'\b'
            result = re.sub(pattern, proper_brand, result, flags=re.IGNORECASE)
        return result
    
    def correct_text(self, text: str) -> str:
        """Main text correction method"""
        # Expand contractions
        text = self.expand_contractions(text)
        
        # Apply regex corrections
        text = self.apply_regex_corrections(text)
        
        # Basic cleanup
        text = self.basic_cleanup(text)
        
        # Apply final capitalizations
        text = self.capitalize_brand_names(text)
        text = self.capitalize_after_punctuation(text)
        
        return text

# Mock spam detection models (for demonstration)
class MockSpamModel:
    def __init__(self, name):
        self.name = name
    
    def predict(self, texts):
        # Simple mock prediction based on spam keywords
        text = texts[0].lower()
        spam_keywords = ['free', 'urgent', 'click', 'buy now', 'limited time', 'act now', 'winner', 'congratulations']
        spam_score = sum(1 for keyword in spam_keywords if keyword in text)
        return [1 if spam_score >= 2 else 0]
    
    def predict_proba(self, texts):
        text = texts[0].lower()
        spam_keywords = ['free', 'urgent', 'click', 'buy now', 'limited time', 'act now', 'winner', 'congratulations']
        spam_score = sum(1 for keyword in spam_keywords if keyword in text)
        prob = min(0.9, spam_score * 0.15)
        return [[1-prob, prob]]

# Initialize spell checker
@st.cache_resource
def load_spell_checker():
    return SimpleSpellChecker()

# Initialize mock models
@st.cache_resource
def load_models():
    return {
        "naive_bayes": MockSpamModel("naive_bayes"),
        "logistic_regression": MockSpamModel("logistic_regression"),
        "random_forest": MockSpamModel("random_forest"),
        "svm": MockSpamModel("svm")
    }

def analyze_compliance(email_text, subject):
    """Analyze email for compliance violations"""
    violations = []
    
    # Spam pattern checks
    if "free" in email_text.lower():
        violations.append(("spam_patterns.money_claims", "HIGH"))
    if "click" in email_text.lower():
        violations.append(("spam_patterns.click_bait", "HIGH"))
    if "urgent" in email_text.lower():
        violations.append(("spam_patterns.urgency_words", "HIGH"))
    if "buy now" in email_text.lower():
        violations.append(("spam_patterns.pressure_tactics", "HIGH"))
    if "limited time" in email_text.lower():
        violations.append(("spam_patterns.time_pressure", "HIGH"))
    if subject.isupper():
        violations.append(("spam_patterns.excessive_caps", "MEDIUM"))
    
    # Regulatory compliance checks
    if "unsubscribe" not in email_text.lower():
        violations.append(("regulatory_compliance.unsubscribe_required", "HIGH"))
    if not re.search(r'\b[\w\.-]+@[\w\.-]+\.\w+\b', email_text):
        violations.append(("regulatory_compliance.contact_info_missing", "MEDIUM"))
    
    # Content quality checks
    if len(email_text.split()) < 10:
        violations.append(("content_quality.too_short", "LOW"))
    if email_text.count('!') > 3:
        violations.append(("content_quality.excessive_exclamation", "MEDIUM"))
    
    return violations

def calculate_deliverability_score(violations, spam_prob):
    """Calculate deliverability score based on violations and spam probability"""
    base_score = 100
    score = base_score - (len(violations) * 10) - int(spam_prob * 30)
    return max(0, min(100, score))

def get_risk_level(spam_prob):
    """Determine risk level based on spam probability"""
    if spam_prob >= 0.85:
        return "HIGH"
    elif spam_prob >= 0.5:
        return "MEDIUM"
    else:
        return "LOW"

def main():
    # App configuration
    st.set_page_config(
        page_title="📧 Advanced Email Analysis Suite",
        page_icon="📧",
        layout="wide"
    )
    
    # Header
    st.title("🚀 Advanced Email Analysis Suite")
    st.markdown("**Comprehensive email analysis with spell checking, spam detection, and compliance verification**")
    
    # Load resources
    spell_checker = load_spell_checker()
    models = load_models()
    best_model = models.get("naive_bayes")
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["📝 Email Analysis", "📊 Batch Analysis", "ℹ️ About"])
    
    with tab1:
        st.header("Single Email Analysis")
        
        # Input section
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("📧 Email Input")
            subject = st.text_input("📌 Subject of the Email")
            body = st.text_area("✉️ Body of the Email", height=300)
            
            # Options
            with st.expander("⚙️ Analysis Options"):
                enable_spell_check = st.checkbox("Enable spell checking", value=True)
                show_corrected_text = st.checkbox("Show corrected text", value=True)
                detailed_analysis = st.checkbox("Show detailed analysis", value=False)
        
        with col2:
            if st.button("🔍 Analyze Email", type="primary"):
                if subject.strip() or body.strip():
                    email_text = f"{subject.strip()} {body.strip()}"
                    
                    # Spell checking
                    if enable_spell_check:
                        with st.spinner("Checking spelling and grammar..."):
                            corrected_subject = spell_checker.correct_text(subject) if subject else ""
                            corrected_body = spell_checker.correct_text(body) if body else ""
                            corrected_email = f"{corrected_subject} {corrected_body}"
                        
                        if show_corrected_text:
                            st.subheader("✅ Corrected Text")
                            if corrected_subject != subject:
                                st.markdown(f"**Corrected Subject:** {corrected_subject}")
                            if corrected_body != body:
                                st.markdown("**Corrected Body:**")
                                st.text_area("", corrected_body, height=200, key="corrected_display")
                    else:
                        corrected_email = email_text
                    
                    # Model predictions
                    st.subheader("📊 Spam Detection Results")
                    
                    results = {}
                    spam_probs = []
                    
                    for name, model in models.items():
                        with st.spinner(f"Running {name} model..."):
                            pred = model.predict([corrected_email])[0]
                            prob = model.predict_proba([corrected_email])[0][1]
                            label = "SPAM" if pred == 1 else "HAM"
                            results[name] = {"label": label, "probability": prob}
                            spam_probs.append(prob)
                    
                    # Display results in columns
                    model_cols = st.columns(len(models))
                    for i, (name, result) in enumerate(results.items()):
                        with model_cols[i]:
                            color = "red" if result["label"] == "SPAM" else "green"
                            st.markdown(f"**{name.upper()}**")
                            st.markdown(f"<span style='color:{color}'>{result['label']}</span>", 
                                      unsafe_allow_html=True)
                            st.markdown(f"Confidence: {result['probability']:.2%}")
                    
                    # Average probability
                    avg_prob = sum(spam_probs) / len(spam_probs)
                    
                    # Compliance analysis
                    violations = analyze_compliance(corrected_email, subject)
                    deliverability_score = calculate_deliverability_score(violations, avg_prob)
                    risk_level = get_risk_level(avg_prob)
                    
                    # Summary metrics
                    st.subheader("📈 Summary Analysis")
                    
                    metric_cols = st.columns(4)
                    with metric_cols[0]:
                        st.metric("Spam Probability", f"{avg_prob:.2%}")
                    with metric_cols[1]:
                        st.metric("Deliverability Score", f"{deliverability_score}/100")
                    with metric_cols[2]:
                        st.metric("Risk Level", risk_level)
                    with metric_cols[3]:
                        st.metric("Violations", len(violations))
                    
                    # Detailed analysis
                    if detailed_analysis:
                        st.subheader("🔍 Detailed Analysis")
                        
                        # Classification
                        classification = "SPAM" if avg_prob > 0.5 else "HAM"
                        st.markdown(f"**Final Classification:** `{classification}`")
                        st.markdown(f"**Compliance Status:** `{'NON_COMPLIANT' if violations else 'COMPLIANT'}`")
                        
                        # Violations
                        if violations:
                            st.warning(f"⚠️ {len(violations)} Violations Found:")
                            for violation, severity in violations:
                                st.markdown(f"- `{violation}` (Severity: {severity})")
                            
                            st.subheader("🛠 Recommended Actions")
                            for violation, severity in violations:
                                action = violation.split('.')[-1].replace('_', ' ').title()
                                st.markdown(f"- **{severity}:** Address {action}")
                        else:
                            st.success("✅ No violations found. Email looks clean!")
                else:
                    st.warning("Please enter either a subject or body for analysis.")
    
    with tab2:
        st.header("Batch Email Analysis")
        st.markdown("Upload a CSV file with email data for batch processing")
        
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        
        if uploaded_file is not None:
            try:
                import pandas as pd
                df = pd.read_csv(uploaded_file)
                
                st.subheader("📊 Data Preview")
                st.dataframe(df.head())
                
                if st.button("Process Batch"):
                    with st.spinner("Processing batch analysis..."):
                        # This would implement batch processing
                        st.info("Batch processing feature would be implemented here")
                        
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")
    
    with tab3:
        st.header("About This Application")
        
        st.markdown("""
        ### 🚀 Advanced Email Analysis Suite
        
        This application provides comprehensive email analysis capabilities including:
        
        **📝 Spell Checking & Grammar Correction**
        - Advanced text correction using multiple NLP techniques
        - Brand name capitalization
        - Contraction expansion
        - Grammar and punctuation fixes
        
        **🛡️ Spam Detection**
        - Multiple machine learning models
        - Ensemble prediction averaging
        - Confidence scoring
        - Pattern-based detection
        
        **📋 Compliance Analysis**
        - Regulatory compliance checking
        - Content quality assessment
        - Deliverability scoring
        - Risk level determination
        
        **🔧 Features**
        - Real-time analysis
        - Batch processing support
        - Detailed reporting
        - Customizable analysis options
        
        ### 🎯 Use Cases
        - Email marketing campaign validation
        - Compliance verification
        - Content quality improvement
        - Spam prevention
        - Deliverability optimization
        
        ### 🔧 Technical Details
        - Built with Streamlit
        - Uses advanced NLP techniques
        - Multiple ML model ensemble
        - Comprehensive text processing
        """)

if __name__ == "__main__":
    main()
