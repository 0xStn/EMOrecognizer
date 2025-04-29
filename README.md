# EMOrecognizer - Text Emotion Detection System

EMOrecognizer is an AI-powered application that analyzes text inputs and predicts the emotions expressed in the text. The system uses machine learning models trained on emotion-labeled text data to classify text into multiple emotional categories.

## Project Overview

This project implements a text-based emotion recognition system with the following features:

- Text preprocessing and cleaning using NLP techniques
- Machine learning models for emotion classification
- Web interface for real-time emotion analysis
- REST API for integration with other applications
- Multiple emotion categories including: anger, happiness, love, sadness, surprise, and more

## Tech Stack

- **Backend**: Python, Flask
- **Machine Learning**: Scikit-learn, NLTK, Pandas
- **Frontend**: HTML, CSS, JavaScript
- **Data Processing**: Pandas, NumPy

## Project Structure

```
EMOrecognizer/
├── app.py                  # Flask application main file
├── data/
│   └── emotion_sentimen_dataset.csv  # Dataset for model training
├── Model/
│   ├── Text Emotion Detection.ipynb  # Notebook for model development
│   ├── text_lr_emotion.pkl           # Trained Logistic Regression model
│   └── text(lr)_emotion.pkl          # Alternative trained model
├── static/
│   └── css/
│       └── style.css       # CSS styling
└── templates/
    └── index.html          # Web interface
```

## Getting Started

### Prerequisites

- Python 3.7 or higher
- pip package manager

### Installation

1. Clone the repository:
   ```
   git clone [your-repository-url]
   cd EMOrecognizer
   ```

2. Create a virtual environment:
   ```
   python -m venv venv
   ```

3. Activate the virtual environment:
   - Windows:
     ```
     venv\Scripts\activate
     ```
   - macOS/Linux:
     ```
     source venv/bin/activate
     ```

4. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

5. Download NLTK resources (if not automatically downloaded):
   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('wordnet')
   nltk.download('stopwords')
   ```

### Running the Application

Run the Flask application:
```
python app.py
```

The application will be available at `http://localhost:5000`.

## Using the Web Interface

1. Navigate to `http://localhost:5000` in your browser
2. Enter text in the input field
3. Select a model (if multiple are available)
4. Submit the form to see the predicted emotion with confidence scores

## Using the API

The application provides a REST API for programmatic access:

**Endpoint**: `/predict` (POST)

**Request Format**:
```json
{
  "text": "your text here",
  "model_type": "logistic_regression" 
}
```

**Response Format**:
```json
{
  "text": "your text here",
  "predicted_emotion": "happiness",
  "emoji": "😊☀️",
  "confidence": 0.85,
  "all_emotions": {
    "anger": 0.05,
    "happiness": 0.85,
    "love": 0.03,
    "sadness": 0.02,
    "worry": 0.05
  },
  "model_used": "logistic_regression"
}
```

## Model Training

The models were trained using the Jupyter notebook in the `Model` directory. The notebook includes:

- Data loading and exploration
- Text preprocessing
- Feature extraction using TF-IDF
- Model training (Logistic Regression, SVM, Random Forest)
- Model evaluation and comparison
- Model serialization for use in the application

## Future Improvements

- Add more sophisticated NLP techniques (word embeddings, transformers)
- Incorporate more emotion categories
- Improve UI/UX with visualizations
- Add multi-language support

## License

This project is licensed under the MIT License - see the LICENSE file for details.