from flask import Flask, request, jsonify, app
from flask_cors import CORS
import numpy as np
import string
import nltk
import joblib
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Initialize
app = Flask(__name__)
CORS(app)

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

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

@app.route('/predict', methods=['POST'])
def predict():
    # Get input data from request
    text = request.json.get('text')

    # NLP
    text = preprocess_text(text)

    # TF-IDF vectorizer
    text_vectorized = tfidf_vectorizer.transform([text]).toarray()

    # Prediction
    predicted_disease = model.predict(text_vectorized)[0]
    predicted_proba = model.predict_proba(text_vectorized)[0]
    score = predicted_proba[np.argmax(predicted_proba)]
    if score > 0.60:
        return jsonify({'predicted': predicted_disease}), 200 # disease
    else:
        return jsonify({'error': 'Please enter valid symptoms'}), 400 # error message

if __name__ == '__main__':
    app.run(debug=True)