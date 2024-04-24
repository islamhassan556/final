from flask import Flask, request, jsonify, app
from flask_cors import CORS
import numpy as np
import string
import nltk
import joblib
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Function to preprocess text
def preprocess_text(text):
    # Convert text to lowercase
    text = text.lower()
    # Remove punctuation
    translator = str.maketrans('', '', string.punctuation)
    text_without_punct = text.translate(translator).strip()
    # Tokenize text
    tokens = word_tokenize(text_without_punct)
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [token for token in tokens if token.isalpha() and token not in stop_words]
    # Lemmatize tokens
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]
    # Join tokens into a string
    preprocessed_text = ' '.join(lemmatized_tokens)
    return preprocessed_text

# Load TF-IDF vectorizer and SVM model
tfidf_vectorizer = joblib.load('tfidf_vectorizer.joblib')
model = joblib.load('best_svm_classifier.joblib')

# Function to predict disease
def predict_disease(text):
    # Preprocess text
    preprocessed_text = preprocess_text(text)
    # Vectorize preprocessed text
    text_vectorized = tfidf_vectorizer.transform([preprocessed_text]).toarray()
    # Predict disease
    predicted_disease = model.predict(text_vectorized)[0]
    # Get predicted probability
    predicted_proba = model.predict_proba(text_vectorized)[0]
    score = predicted_proba[np.argmax(predicted_proba)]
    # Return prediction result and status code
    if score > 0.60:
        return {'predicted': predicted_disease}, 200  # Disease
    else:
        return {'error': 'Please enter valid symptoms'}, 400  # Error message

# Route for prediction endpoint
@app.route('/predict', methods=['POST'])
def predict_endpoint():
    # Get input text from request
    input_text = request.json.get('text')
    # Predict disease
    result, status_code = predict_disease(input_text)
    # Return prediction result and status code
    return jsonify(result), status_code

if __name__ == '__main__':
    # Run Flask app
    app.run(debug=True)
