# emails# Texpedition

Texpedition is a comprehensive text processing toolkit that combines advanced NLP techniques, machine learning, and text analysis to provide powerful text correction, normalization, and spam detection capabilities.

## Features

### Text Processing and Correction
- **Enhanced Spell Checking**: Combines multiple NLP methods for accurate text correction
- **Context-Aware Corrections**: Uses BERT models to understand context for better corrections
- **Brand Name Standardization**: Automatically corrects and standardizes brand name capitalization
- **Contraction Handling**: Expands and normalizes contractions in text
- **Word Segmentation**: Fixes concatenated words (e.g., "helloworld" → "hello world")
- **Pattern-Based Corrections**: Uses regular expressions to fix common text errors

### Spam Detection
- **Machine Learning Classification**: Uses multiple models (Naive Bayes, Logistic Regression, Random Forest)
- **Compliance Checking**: Analyzes emails for regulatory compliance with laws like CAN-SPAM and GDPR
- **Content Quality Analysis**: Identifies issues that affect email deliverability
- **Comprehensive Reporting**: Provides detailed analysis with actionable recommendations
- **Bulk Analysis**: Supports processing multiple emails with summary reporting

### Performance Optimization
- **Hardware Acceleration**: Supports CUDA (NVIDIA GPUs) and MPS (Apple Silicon) for faster processing
- **Efficient Data Handling**: Uses optimized data structures and serialization

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/texpedition.git
cd texpedition

# Install dependencies
pip install -r requirements.txt

# Download required spaCy model
python -m spacy download en_core_web_sm

# Download NLTK data (optional, for enhanced dictionary)
python -c "import nltk; nltk.download('words'); nltk.download('wordnet')"
```

## Dependencies

- **NLP and Text Processing**:
  - spacy: Natural language processing
  - enchant: Spell checking
  - contextualSpellCheck: Context-aware spell checking
  - textblob: Additional text processing
  - transformers: BERT models
  - contractions: Handling contractions
  - wordsegment: Word segmentation
  - nltk: Natural language toolkit

- **Machine Learning**:
  - torch: Deep learning with GPU/MPS support
  - pandas: Data manipulation
  - scikit-learn: Machine learning algorithms
  - joblib: Model serialization

- **Web Scraping**:
  - beautifulsoup4: HTML parsing

## Usage Examples

### Text Correction

```python
from main import EnhancedSpellChecker

# Initialize the spell checker
checker = EnhancedSpellChecker(use_bert=True, use_spellcheck=True)

# Sample text with errors
text = """airbnb
Hi Matthew,
Thanks for using airbnb. We really appreciate you choosing airbnb for your travel
plans.
To help us improve, we'd like to ask you a few questions about your experience
so far. itll only take 3minutes, and youranswers will help us make airbnb even
better for you and other guests."""

# Correct the text
corrected_text = checker.correct_text(text)
print(corrected_text)
```

### Spam Detection

```python
from spam import SpamDetector, EnhancedSpamDetectionAPI

# Initialize the spam detector
detector = SpamDetector(use_mps=True)  # Use Apple Silicon acceleration if available

# Load pre-trained models
detector.load_models('spam_detector_models.pkl')

# Initialize the API
api = EnhancedSpamDetectionAPI(detector)

# Analyze an email
email_content = """
Hi there,
Thank you for your recent purchase. Your order will be shipped within 2 business days.
For any questions, please contact our support team.
Best regards,
Customer Service
"""

# Get comprehensive analysis
report = api.analyze_email(
    email_content,
    subject_line="Order Confirmation",
    sender_info="support@company.com"
)

# Print results
print(f"Spam Classification: {report['spam_detection']['classification']}")
print(f"Compliance Status: {report['overall_assessment']['compliance_status']}")
print(f"Deliverability Score: {report['overall_assessment']['deliverability_score']}/100")
```

## Data Files

- **text_processing_data.json**: Human-readable data for text processing
- **text_processing_data.pkl**: Serialized data for efficient loading
- **spam_detector_models.pkl**: Trained spam detection models

## License

[MIT License](LICENSE)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
