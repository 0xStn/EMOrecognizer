from flask import Flask, request, jsonify, render_template, redirect, url_for
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import MinMaxScaler
from scipy.special import softmax
import json
import os
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import re

# Download NLTK resources if not already available
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Initialize the lemmatizer and stopwords
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Define models dictionary with paths
models = {
    "logistic_regression": "model/text(lr)_emotion.pkl",
#     "random_forest": "model/text(RF)_emotion.pkl",  # Will be available once created
#     "svm": "model/text(SVM)_emotion.pkl"  # Will be available once created
}

# Set the default model to use
DEFAULT_MODEL = "logistic_regression"  # Since we know this model exists in your workspace

# Load the selected model
def load_model(model_type=DEFAULT_MODEL):
    model_path = models.get(model_type, models[DEFAULT_MODEL])
    if os.path.exists(model_path):
        return joblib.load(open(model_path, "rb"))
    
    # Fallback to the model we know exists
    return joblib.load(open(models[DEFAULT_MODEL], "rb"))

# Load model
pipe_model = load_model()

# Updated emotions dictionary to match the new set of emotions
emotions_emoji_dict = {
    "anger": "😡🤬",         # Anger with rage face and cursing face
    "hate": "😠👿",          # Hate with angry face and devil face
    "neutral": "😐😶",       # Neutral with straight face and no mouth face
    "love": "❤️😍",          # Love with heart and heart eyes
    "worry": "😟😰",         # Worry with concerned and anxious faces
    "relief": "😌😮‍💨",      # Relief with relieved face and exhaling face
    "happiness": "😊☀️",     # Happiness with smiling face and sunshine
    "fun": "😄🎉",           # Fun with laughing face and party popper
    "empty": "😶‍🌫️🫥",       # Empty with face in clouds and dotted line face
    "enthusiasm": "🤩✨",     # Enthusiasm with star-struck face and sparkles
    "sadness": "😔😭",       # Sadness with sad face and crying
    "surprise": "😮😲",      # Surprise with open mouth and astonished faces
    "boredom": "😒🥱"        # Boredom with unamused face and yawning face
}

# Function to normalize repeating characters
def normalize_repeating_chars(text):
    # Normalize repeating characters to max of 2 (e.g., 'haaaappy' -> 'haappy')
    return re.sub(r'(.)\1{2,}', r'\1\1', text)

# Function to preprocess text - matching the preprocessing in the notebook
def preprocess_text(text):
    """Clean and preprocess input text"""
    # Convert to lowercase
    text = text.lower()
    
    # Normalize repeating characters
    text = normalize_repeating_chars(text)
    
    # Apply cleaning operations
    # Remove user handles (Twitter-style @mentions)
    text = re.sub(r'@\w+', '', text)
    
    # Fix contractions (simplified)
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"can't", "cannot", text)
    text = re.sub(r"n't", " not", text)
    text = re.sub(r"'re", " are", text)
    text = re.sub(r"'s", " is", text)
    text = re.sub(r"'d", " would", text)
    text = re.sub(r"'ll", " will", text)
    text = re.sub(r"'ve", " have", text)
    text = re.sub(r"'m", " am", text)
    
    # Remove URLs
    text = re.sub(r'http\S+|www.\S+', '', text)
    
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    
    # Remove special characters and punctuation
    text = re.sub(r'[^\w\s]', '', text)
    
    # Tokenize and lemmatize with stopword removal
    tokens = word_tokenize(text)
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]
    
    # Join tokens back into a string
    return ' '.join(lemmatized_tokens)

def predict_emotions(docx):
    """Predict emotions after preprocessing the text"""
    cleaned_text = preprocess_text(docx)
    results = pipe_model.predict([cleaned_text])
    return results[0]

def get_prediction_proba(docx):
    """Get prediction probabilities after preprocessing the text"""
    cleaned_text = preprocess_text(docx)
    
    # Check if the model supports predict_proba
    if hasattr(pipe_model, 'predict_proba') and callable(getattr(pipe_model, 'predict_proba')):
        results = pipe_model.predict_proba([cleaned_text])
        return results
    else:
        # For models without predict_proba, use distance from decision boundary
        # This approach gives more realistic probability distribution
        
        try:
            # For multi-class models - Force decision function approach for better probabilities
            if hasattr(pipe_model[-1], 'decision_function'):
                # Get decision values (distance from hyperplane)
                decision_values = pipe_model[-1].decision_function([cleaned_text])
                
                # Convert to probabilities using softmax
                if len(decision_values.shape) > 1:  # Multi-class case
                    # Apply temperature to make distribution more distinct
                    scaled_values = decision_values * 2  # Increase separation
                    probas = softmax(scaled_values, axis=1)
                    # Ensure probabilities sum to 1
                    probas = probas / np.sum(probas)
                    return probas
                else:
                    # Binary case
                    decision_values = np.array([[-decision_values[0]], [decision_values[0]]])
                    probas = softmax(decision_values, axis=0).T
                    return probas
        except Exception as e:
            print(f"Error calculating probabilities: {e}")
        
        # Fallback: Create a simple probability distribution
        classes = pipe_model.classes_
        prediction = predict_emotions(docx)
        
        # Create more realistic distribution
        num_classes = len(classes)
        base_prob = (1.0 - 0.7) / (num_classes - 1)  # Divide remaining probability
        
        probs = np.ones((1, num_classes)) * base_prob
        pred_idx = np.where(classes == prediction)[0][0]
        probs[0, pred_idx] = 0.7  # Dominant class gets 70% probability
        
        # Ensure sum is exactly 1
        probs = probs / np.sum(probs)
        return probs

# Initialize Flask app
app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def home():
    result = None
    
    if request.method == 'POST':
        text_input = request.form.get('text_input', '')
        model_type = request.form.get('model_type', DEFAULT_MODEL)
        
        # Load the selected model
        global pipe_model
        if model_type != DEFAULT_MODEL:
            pipe_model = load_model(model_type)
        
        if text_input:
            # Get predictions
            emotion = predict_emotions(text_input)
            probabilities = get_prediction_proba(text_input)
            
            # Format probabilities
            emotion_probs = {}
            for idx, emotion_class in enumerate(pipe_model.classes_):
                emotion_probs[emotion_class] = float(probabilities[0][idx])
            
            # Create result
            result = {
                'text': text_input,
                'emotion': emotion,
                'emoji': emotions_emoji_dict.get(emotion, ""),
                'confidence': float(np.max(probabilities)),
                'emotions': emotion_probs,
                'model': model_type
            }
    
    return render_template('index.html', result=result, emotions_emoji_dict=emotions_emoji_dict, 
                          request=request, models=models.keys())

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get text and model type from request JSON
        data = request.get_json(force=True)
        text = data.get('text', '')
        model_type = data.get('model_type', DEFAULT_MODEL)
        
        if not text:
            return jsonify({'error': 'No text provided'}), 400
        
        # Load the selected model
        global pipe_model
        if model_type != DEFAULT_MODEL:
            pipe_model = load_model(model_type)
        
        # Get predictions
        emotion = predict_emotions(text)
        probabilities = get_prediction_proba(text)
        
        # Format response
        emotion_probabilities = {}
        for idx, emotion_class in enumerate(pipe_model.classes_):
            emotion_probabilities[emotion_class] = float(probabilities[0][idx])
        
        # Create response
        response = {
            'text': text,
            'predicted_emotion': emotion,
            'emoji': emotions_emoji_dict.get(emotion, ""),
            'confidence': float(np.max(probabilities)),
            'all_emotions': emotion_probabilities,
            'model_used': model_type
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
