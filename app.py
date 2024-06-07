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
# Greeting detection function
def detect_greeting(user_input):
    english_greetings = [
        "hi", "hello", "hey", "howdy", "greetings", "good morning", "good afternoon",
        "good evening", "introduce yourself", "what is your job", "who are you",
        "tell me what you offer", "what are your services", "what's up", "hiya",
        "how are you", "how is it going", "what's new", "what's happening", "salutations",
        "good day", "yo", "how's it going", "how's everything", "how are things", 
        "howdy-do", "how have you been", "long time no see", "nice to see you", 
        "good to see you", "pleased to meet you", "how's your day", "how's your day going",
        "how's life", "what's going on", "how's it hanging", "how's tricks", 
        "what's the good word", "what's the news", "top of the morning to you", 
        "how do you do", "look who it is", "what's crackin'", "what's cooking", 
        "what's shaking", "what's sizzling", "what's happening", "how's everything going",
        "what's the latest", "what's new", "what's the word", "yo yo", "hey there", 
        "hi there", "what's going down", "hey, how's it going", "hello there", 
        "how are you doing", "hey, what’s new", "hey, how are things", "how have things been", 
        "how’s it going with you", "how’s everything been", "how’s life treating you", 
        "how’s your day been", "hey, how’s everything", "hello, what’s up", "hi, how’s it going",
        "g'day", "hey, how’ve you been"
    ]
    
    arabic_greetings = [
        "مرحبا", "مرحبًا", "أهلا", "أهلًا", "مساء الخير", "صباح الخير", "عرف نفسك",
        "ما هي وظيفتك", "من انت؟", "من انت", "عرفني بنفسك", "اخبرني ماذا تقدم",
        "ما هي خدماتك", "كيف حالك", "ما الأخبار", "مرحبتين", "السلام عليكم", "وعليكم السلام",
        "تحية", "مساء النور", "صباح النور", "كيف حالك اليوم", "كيف حالك الآن", "أهلاً وسهلاً",
        "مرحبا بك", "يسعد صباحك", "يسعد مساك", "شخبارك", "شلونك", "كيفك", "كيف أصبحت",
        "كيف أمسيت", "كيف الحال", "كيف الأمور", "كيف الدنيا", "شو الأخبار", "كيف الوضع",
        "إزيك", "إزي الحال", "كيف الحالك", "كيف حالك اليوم", "إزيك يا حلو", "شو مسوي",
        "شو عامل", "شو أخبارك", "كيف الأمور معك", "شو الوضع", "شو الأخبار", "أهلاً بك",
        "أهلاً بعودتك", "مساء الورد", "صباح الورد", "مساء الفل", "صباح الفل", "صباح الورد والياسمين",
        "مساء الورد والياسمين", "كيف الأمور عندك", "كيف الدنيا معك", "أهلاً عزيزي", "مرحبا عزيزي",
        "تحياتي لك", "أهلاً وسهلاً بك", "أهلاً وسهلاً بك يا", "شو عامل اليوم", "شو مسوي اليوم",
        "كيف كانت يومك", "كيف كانت ليلتك", "كيف هي أحوالك"
    ]
    for greeting in english_greetings:
        if greeting.lower() in user_input.lower():
            return True
    for greeting in arabic_greetings:
        if greeting in user_input:
            return True
    return False


# Greeting response
def respond_to_greeting(lang):
    if lang == 'ar':
        return  "مرحبًا! أنا مساعد الرعاية الصحية الخاص بك! يرجى إدخال الأعراض الخاصة بك، وسأبذل قصارى جهدي لإخبارك عن أمراضك والاحتياطات الموصى بها لمساعدتك على حماية نفسك"
    else:
        return  "Hello! I'm your healthcare assistant!, Please enter your symptoms, and I'll do my best to tell your diseases and recommended precautions to help you protect yourself"

@app.route('/predict', methods=['POST'])
def predict():
    # Get input data from request
    text = request.json.get('text')

    # Detect input language
    translator = Translator()
    lang = translator.detect(text).lang

    # Check for greetings
    if detect_greeting(text):
        return jsonify({'predicted': respond_to_greeting(lang)}), 200

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
