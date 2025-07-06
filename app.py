import streamlit as st
import joblib
import re
import logging
from typing import Optional, Tuple, Set, List
import warnings
import pickle
import os
from datetime import datetime

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Enhanced Text Processing & Spam Detection",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .section-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2e8b57;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    
    .result-box {
        background-color: #f0f8ff;
        border: 2px solid #1f77b4;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
    }
    
    .spam-warning {
        background-color: #ffe4e1;
        border: 2px solid #dc143c;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
    }
    
    .ham-safe {
        background-color: #f0fff0;
        border: 2px solid #32cd32;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
    }
    
    .stats-box {
        background-color: #f5f5f5;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'processing_history' not in st.session_state:
    st.session_state.processing_history = []

# Simplified spell checker class for Streamlit
class StreamlitSpellChecker:
    """
    Simplified spell checker for Streamlit deployment.
    This version focuses on core functionality with minimal dependencies.
    """
    
    def __init__(self):
        self.brand_names = {
            'airbnb': 'Airbnb',
            'google': 'Google',
            'facebook': 'Facebook',
            'microsoft': 'Microsoft',
            'amazon': 'Amazon',
            'apple': 'Apple',
            'netflix': 'Netflix',
            'spotify': 'Spotify',
            'youtube': 'YouTube',
            'twitter': 'Twitter',
            'instagram': 'Instagram',
            'linkedin': 'LinkedIn',
            'whatsapp': 'WhatsApp',
            'telegram': 'Telegram',
            'zoom': 'Zoom',
            'skype': 'Skype',
            'paypal': 'PayPal',
            'uber': 'Uber',
            'lyft': 'Lyft',
            'tesla': 'Tesla',
            'samsung': 'Samsung',
            'sony': 'Sony',
            'nike': 'Nike',
            'adidas': 'Adidas',
            'mcdonalds': "McDonald's",
            'starbucks': 'Starbucks',
            'walmart': 'Walmart',
            'target': 'Target',
            'costco': 'Costco',
            'ikea': 'IKEA',
            'visa': 'Visa',
            'mastercard': 'Mastercard',
            'amex': 'American Express',
            'fedex': 'FedEx',
            'ups': 'UPS',
            'dhl': 'DHL',
            'boeing': 'Boeing',
            'airbus': 'Airbus',
            'bmw': 'BMW',
            'mercedes': 'Mercedes',
            'audi': 'Audi',
            'volkswagen': 'Volkswagen',
            'toyota': 'Toyota',
            'honda': 'Honda',
            'ford': 'Ford',
            'chevrolet': 'Chevrolet',
            'hyundai': 'Hyundai',
            'kia': 'Kia',
            'nissan': 'Nissan',
            'subaru': 'Subaru',
            'mazda': 'Mazda',
            'volvo': 'Volvo',
            'jaguar': 'Jaguar',
            'bentley': 'Bentley',
            'ferrari': 'Ferrari',
            'lamborghini': 'Lamborghini',
            'maserati': 'Maserati',
            'porsche': 'Porsche',
            'bugatti': 'Bugatti',
            'rolls-royce': 'Rolls-Royce',
            'aston martin': 'Aston Martin',
            'mclaren': 'McLaren',
            'lotus': 'Lotus',
            'koenigsegg': 'Koenigsegg',
            'pagani': 'Pagani',
            'bugatti': 'Bugatti',
            'maybach': 'Maybach',
            'infiniti': 'Infiniti',
            'lexus': 'Lexus',
            'acura': 'Acura',
            'cadillac': 'Cadillac',
            'lincoln': 'Lincoln',
            'buick': 'Buick',
            'gmc': 'GMC',
            'jeep': 'Jeep',
            'ram': 'RAM',
            'dodge': 'Dodge',
            'chrysler': 'Chrysler',
            'mitsubishi': 'Mitsubishi',
            'suzuki': 'Suzuki',
            'isuzu': 'Isuzu',
            'peugeot': 'Peugeot',
            'citroen': 'Citroën',
            'renault': 'Renault',
            'fiat': 'Fiat',
            'alfa romeo': 'Alfa Romeo',
            'lancia': 'Lancia',
            'ferrari': 'Ferrari',
            'lamborghini': 'Lamborghini',
            'maserati': 'Maserati',
            'pagani': 'Pagani',
            'koenigsegg': 'Koenigsegg',
            'mclaren': 'McLaren',
            'lotus': 'Lotus',
            'aston martin': 'Aston Martin',
            'bentley': 'Bentley',
            'rolls-royce': 'Rolls-Royce',
            'bugatti': 'Bugatti',
            'maybach': 'Maybach'
        }
        
        self.manual_contractions = {
            "dont": "don't",
            "wont": "won't",
            "cant": "can't",
            "isnt": "isn't",
            "arent": "aren't",
            "wasnt": "wasn't",
            "werent": "weren't",
            "hasnt": "hasn't",
            "havent": "haven't",
            "hadnt": "hadn't",
            "shouldnt": "shouldn't",
            "wouldnt": "wouldn't",
            "couldnt": "couldn't",
            "mustnt": "mustn't",
            "neednt": "needn't",
            "darent": "daren't",
            "oughtnt": "oughtn't",
            "shant": "shan't",
            "theyre": "they're",
            "were": "we're",
            "youre": "you're",
            "its": "it's",
            "thats": "that's",
            "whats": "what's",
            "wheres": "where's",
            "whos": "who's",
            "hows": "how's",
            "whys": "why's",
            "whens": "when's",
            "theres": "there's",
            "heres": "here's",
            "ive": "I've",
            "youve": "you've",
            "weve": "we've",
            "theyve": "they've",
            "ill": "I'll",
            "youll": "you'll",
            "hell": "he'll",
            "shell": "she'll",
            "well": "we'll",
            "theyll": "they'll",
            "itll": "it'll",
            "thatll": "that'll",
            "im": "I'm",
            "id": "I'd",
            "youd": "you'd",
            "hed": "he'd",
            "shed": "she'd",
            "wed": "we'd",
            "theyd": "they'd",
            "itd": "it'd",
            "thatd": "that'd"
        }
        
        self.corrections = [
            (r'\b(\d+)([a-zA-Z])', r'\1 \2'),  # Fix spacing between numbers and letters
            (r'([a-z])([A-Z]{2,})', r'\1 \2'),  # Fix spacing between lowercase and uppercase
            (r'([.!?,:;])([a-zA-Z])', r'\1 \2'),  # Fix spacing after punctuation
            (r'\s+', ' '),  # Normalize whitespace
        ]
    
    def expand_contractions(self, text: str) -> str:
        """Expand contractions using manual patterns"""
        for contraction, expansion in self.manual_contractions.items():
            text = re.sub(r'\b' + re.escape(contraction) + r'\b', expansion, text, flags=re.IGNORECASE)
        return text
    
    def apply_regex_corrections(self, text: str) -> str:
        """Apply regex-based corrections"""
        for pattern, replacement in self.corrections:
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        return text
    
    def capitalize_brand_names(self, text: str) -> str:
        """Capitalize brand names"""
        for lowercase_brand, proper_brand in self.brand_names.items():
            pattern = r'\b' + re.escape(lowercase_brand) + r'\b'
            text = re.sub(pattern, proper_brand, text, flags=re.IGNORECASE)
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
    
    def correct_text(self, text: str) -> str:
        """Main text correction method"""
        # Expand contractions
        text = self.expand_contractions(text)
        
        # Apply regex corrections
        text = self.apply_regex_corrections(text)
        
        # Handle special characters
        text = text.replace('·', ' ').replace('\u00b7', ' ')
        
        # Capitalize brand names
        text = self.capitalize_brand_names(text)
        
        # Capitalize after punctuation
        text = self.capitalize_after_punctuation(text)
        
        return text

# Load spam detection models
@st.cache_resource
def load_spam_models():
    """Load spam detection models with error handling"""
    try:
        if os.path.exists('spam_detector_models.pkl'):
            models = joblib.load('spam_detector_models.pkl')
            return models
        else:
            st.warning("⚠️ Spam detection models not found. Please ensure 'spam_detector_models.pkl' is in the same directory.")
            return None
    except Exception as e:
        st.error(f"❌ Error loading spam models: {str(e)}")
        return None

# Initialize spell checker
@st.cache_resource
def load_spell_checker():
    """Initialize spell checker"""
    return StreamlitSpellChecker()

def classify_spam(text: str, models: dict) -> dict:
    """Classify text as spam or ham using all available models"""
    if not models:
        return {}
    
    results = {}
    try:
        for name, model in models.items():
            pred = model.predict([text])[0]
            probability = model.predict_proba([text])[0] if hasattr(model, 'predict_proba') else None
            results[name] = {
                'prediction': pred,
                'label': "SPAM" if pred == 1 else "HAM",
                'probability': probability
            }
    except Exception as e:
        st.error(f"❌ Error in spam classification: {str(e)}")
    
    return results

def get_consensus_prediction(results: dict) -> tuple:
    """Get consensus prediction from all models"""
    if not results:
        return None, None
    
    spam_count = sum(1 for r in results.values() if r['prediction'] == 1)
    total_models = len(results)
    
    is_spam = spam_count > total_models / 2
    confidence = spam_count / total_models if is_spam else (total_models - spam_count) / total_models
    
    return is_spam, confidence

def main():
    """Main Streamlit application"""
    
    # Header
    st.markdown('<div class="main-header">📝 Enhanced Text Processing & Spam Detection</div>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("⚙️ Settings")
    
    # Processing options
    st.sidebar.subheader("Processing Options")
    enable_spell_check = st.sidebar.checkbox("Enable Spell Check", value=True)
    enable_spam_detection = st.sidebar.checkbox("Enable Spam Detection", value=True)
    
    # Load models and checker
    spell_checker = load_spell_checker()
    spam_models = load_spam_models() if enable_spam_detection else None
    
    # Model status
    st.sidebar.subheader("📊 Model Status")
    st.sidebar.success("✅ Spell Checker: Ready")
    if spam_models:
        st.sidebar.success(f"✅ Spam Models: {len(spam_models)} loaded")
        st.sidebar.write("**Available Models:**")
        for model_name in spam_models.keys():
            st.sidebar.write(f"• {model_name}")
    else:
        st.sidebar.warning("⚠️ Spam Models: Not available")
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown('<div class="section-header">📝 Input Text</div>', unsafe_allow_html=True)
        
        # Text input options
        input_method = st.radio("Choose input method:", ["Text Area", "File Upload"])
        
        if input_method == "Text Area":
            input_text = st.text_area(
                "Enter your text here:",
                height=300,
                placeholder="Type or paste your text here...",
                help="Enter the text you want to process for spell checking and spam detection."
            )
        else:
            uploaded_file = st.file_uploader(
                "Upload a text file",
                type=['txt'],
                help="Upload a .txt file to process"
            )
            
            if uploaded_file is not None:
                input_text = str(uploaded_file.read(), "utf-8")
                st.text_area("File content:", value=input_text, height=200, disabled=True)
            else:
                input_text = ""
        
        # Processing button
        process_button = st.button("🚀 Process Text", type="primary", use_container_width=True)
    
    with col2:
        st.markdown('<div class="section-header">📊 Results</div>', unsafe_allow_html=True)
        
        if process_button and input_text.strip():
            # Process text
            with st.spinner("Processing text..."):
                processed_text = input_text
                spell_check_applied = False
                spam_results = {}
                
                # Apply spell checking
                if enable_spell_check:
                    processed_text = spell_checker.correct_text(input_text)
                    spell_check_applied = True
                
                # Apply spam detection
                if enable_spam_detection and spam_models:
                    spam_results = classify_spam(processed_text, spam_models)
                
                # Store in history
                st.session_state.processing_history.append({
                    'timestamp': datetime.now(),
                    'original': input_text,
                    'processed': processed_text,
                    'spell_checked': spell_check_applied,
                    'spam_results': spam_results
                })
            
            # Display results
            if spell_check_applied:
                st.markdown("### ✏️ Spell Checked Text")
                st.markdown('<div class="result-box">', unsafe_allow_html=True)
                st.text_area("Corrected text:", value=processed_text, height=200, disabled=True)
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Show changes
                if processed_text != input_text:
                    st.markdown("### 🔄 Changes Made")
                    st.success("✅ Text has been corrected!")
                    
                    # Basic diff visualization
                    with st.expander("📋 View Changes"):
                        st.markdown("**Original:**")
                        st.code(input_text, language=None)
                        st.markdown("**Corrected:**")
                        st.code(processed_text, language=None)
                else:
                    st.info("ℹ️ No spelling corrections needed!")
            
            # Display spam detection results
            if spam_results:
                st.markdown("### 🛡️ Spam Detection Results")
                
                # Get consensus
                is_spam, confidence = get_consensus_prediction(spam_results)
                
                if is_spam:
                    st.markdown('<div class="spam-warning">', unsafe_allow_html=True)
                    st.error(f"🚨 **SPAM DETECTED** (Confidence: {confidence:.1%})")
                    st.markdown('</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="ham-safe">', unsafe_allow_html=True)
                    st.success(f"✅ **LEGITIMATE MESSAGE** (Confidence: {confidence:.1%})")
                    st.markdown('</div>', unsafe_allow_html=True)
                
                # Detailed results
                with st.expander("📊 Detailed Model Predictions"):
                    for model_name, result in spam_results.items():
                        col_model, col_pred, col_prob = st.columns([2, 1, 1])
                        
                        with col_model:
                            st.write(f"**{model_name}**")
                        
                        with col_pred:
                            if result['label'] == 'SPAM':
                                st.error("SPAM")
                            else:
                                st.success("HAM")
                        
                        with col_prob:
                            if result['probability'] is not None:
                                spam_prob = result['probability'][1] if len(result['probability']) > 1 else result['probability'][0]
                                st.write(f"{spam_prob:.3f}")
                            else:
                                st.write("N/A")
        
        elif process_button:
            st.warning("⚠️ Please enter some text to process!")
    
    # Statistics and history
    if st.session_state.processing_history:
        st.markdown("---")
        st.markdown('<div class="section-header">📈 Processing Statistics</div>', unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        
        total_processed = len(st.session_state.processing_history)
        spell_checked = sum(1 for h in st.session_state.processing_history if h['spell_checked'])
        spam_detected = sum(1 for h in st.session_state.processing_history 
                          if h['spam_results'] and get_consensus_prediction(h['spam_results'])[0])
        
        with col1:
            st.metric("Total Processed", total_processed)
        
        with col2:
            st.metric("Spell Checked", spell_checked)
        
        with col3:
            st.metric("Spam Detected", spam_detected)
        
        with col4:
            st.metric("Legitimate Messages", total_processed - spam_detected)
        
        # Processing history
        if st.expander("📋 Processing History"):
            for i, entry in enumerate(reversed(st.session_state.processing_history[-10:])):  # Show last 10
                st.markdown(f"**#{total_processed - i}** - {entry['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.text_area(
                        f"Original #{total_processed - i}:", 
                        value=entry['original'][:200] + "..." if len(entry['original']) > 200 else entry['original'],
                        height=100,
                        disabled=True,
                        key=f"orig_{i}"
                    )
                
                with col2:
                    if entry['spell_checked']:
                        st.text_area(
                            f"Processed #{total_processed - i}:", 
                            value=entry['processed'][:200] + "..." if len(entry['processed']) > 200 else entry['processed'],
                            height=100,
                            disabled=True,
                            key=f"proc_{i}"
                        )
                    
                    if entry['spam_results']:
                        is_spam, confidence = get_consensus_prediction(entry['spam_results'])
                        if is_spam:
                            st.error(f"🚨 SPAM ({confidence:.1%})")
                        else:
                            st.success(f"✅ HAM ({confidence:.1%})")
                
                st.markdown("---")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; margin-top: 2rem;">
        <p>📝 Enhanced Text Processing & Spam Detection System</p>
        <p>Built with Streamlit • Powered by Machine Learning</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
