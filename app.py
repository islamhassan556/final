from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import string
import nltk
import joblib
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from googletrans import Translator

# Initialize
app = Flask(__name__)
CORS(app)

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Translation functions
def translate_to_arabic(text):
    translator = Translator()
    translation = translator.translate(text, src='en', dest='ar')
    return translation.text

def translate_to_english(text):
    translator = Translator()
    translation = translator.translate(text, src='ar', dest='en')
    return translation.text

# NLP
def lowercase_text(text):
    return text.lower()

def remove_punctuation(text):
    translator = str.maketrans('', '', string.punctuation)
    text_without_punct = text.translate(translator).strip()
    return text_without_punct

def tokenize_text(text):
    tokens = word_tokenize(text)
    return tokens

def remove_stopwords(tokens):
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [token for token in tokens if token.isalpha() and token not in stop_words]
    return filtered_tokens

def lemmatize_text(tokens):
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return lemmatized_tokens

# NLP Container function
def preprocess_text(text):
    text = lowercase_text(text)
    text = remove_punctuation(text)
    tokens = tokenize_text(text)
    tokens = remove_stopwords(tokens)
    tokens = lemmatize_text(tokens)
    preprocessed_text = ' '.join(tokens)
    return preprocessed_text

# Load the TF-IDF vectorizer
tfidf_vectorizer = joblib.load('tfidf_vectorizer.joblib')

# Load the SVM model
model = joblib.load('best_svm_classifier.joblib')

# Greeting detection function
def detect_greeting(user_input):
    english_greetings = ["hi", "hello", "hey", "howdy", "greetings", "what's up", "good morning", "good afternoon", "good evening", "introduce yourself"]
    arabic_greetings = ["مرحبا", "مرحبًا", "أهلا", "أهلًا", "مساء الخير", "صباح الخير"]
    for word in user_input.split():
        if word.lower() in english_greetings or word in arabic_greetings:
            return True
    return False

# Greeting response
def respond_to_greeting(lang):
    if lang == 'ar':
        return "مرحبًا! أنا مساعدك الصحي. كيف يمكنني مساعدتك اليوم؟"
    else:
        return "Hello! I'm your healthcare assistant. How can I assist you today?"

@app.route('/predict', methods=['POST'])
def predict():
    # Get input data from request
    text = request.json.get('text')

    # Detect input language
    translator = Translator()
    lang = translator.detect(text).lang

    # Check for greetings
    if detect_greeting(text):
        return jsonify({'response': respond_to_greeting(lang)}), 200

    # Translate input to English if it's in Arabic
    if lang == 'ar':
        translated_text = translate_to_english(text)
    else:
        translated_text = text

    # NLP
    preprocessed_text = preprocess_text(translated_text)

    # TF-IDF vectorizer
    text_vectorized = tfidf_vectorizer.transform([preprocessed_text]).toarray()

    # Prediction
    predicted_disease = model.predict(text_vectorized)[0]
    predicted_proba = model.predict_proba(text_vectorized)[0]
    score = predicted_proba[np.argmax(predicted_proba)]

    if score > 0.30:
        # Translate predicted disease to the input language
        if lang == 'ar':
            translated_disease = translate_to_arabic(predicted_disease)
            response = {'predicted': "من المحتمل أنك تعاني من " + translated_disease}
        else:
            response = {'predicted': f"Maybe you suffer from {predicted_disease}"}
    
        return jsonify(response), 200
    else:
        if lang == 'ar':
            response = {'error': 'الرجاء إدخال أعراض صحيحة'}
        else:
            response = {'error': 'Please enter valid symptoms'}
    
        return jsonify(response), 400

if __name__ == '__main__':
    app.run(debug=True)
