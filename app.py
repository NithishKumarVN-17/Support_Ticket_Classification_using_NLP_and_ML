from flask import Flask, render_template, request
import pickle
import re
import numpy as np
from textblob import TextBlob
import spacy
from typing import Optional, Dict

app = Flask(__name__)

# Load the models
with open(r'Models/word2vec_model.pkl', 'rb') as f:
    word2vec_model = pickle.load(f)

with open(r'Models/support_classification_model.pkl', 'rb') as f:
    support_classification_model = pickle.load(f)

# Classification types mapping
classification_types: Dict[int, str] = { 
    0: "Bank account services",
    1: "Credit Card/Prepaid Card",
    2: "Mortgages/Loans",
    3: "Theft/Dispute reporting",
    4: "Others"
}

class TextProcessor:
    def __init__(self, word2vec_model: any) -> None:
        """
        Initializes the TextProcessor with a Word2Vec model.

        :param word2vec_model: The Word2Vec model to use for generating word vectors.
        """
        self.word2vec_model = word2vec_model
        self.nlp = spacy.load('en_core_web_sm')

    def clean_text(self, text: str) -> str:
        """
        Cleans the input text by converting to lowercase and removing unwanted characters.

        :param text: The input text to clean.
        :return: The cleaned text.
        """
        text = text.lower()
        text = re.sub(r'\[.*?\]', '', text)
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\b\w*\d\w*\b', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def lemmatize_text(self, text: str) -> str:
        """
        Lemmatizes the input text using spaCy.

        :param text: The input text to lemmatize.
        :return: The lemmatized text.
        """
        doc = self.nlp(text)
        return ' '.join([token.lemma_ for token in doc])

    def get_POS_tags(self, text: str) -> str:
        """
        Extracts relevant parts of speech (nouns, adjectives, verbs) from the input text.

        :param text: The input text from which to extract POS tags.
        :return: A string of relevant words based on the POS tags.
        """
        tags_required = {'NN', 'JJ', 'VB'}
        blob = TextBlob(text)
        return ' '.join([word for word, tag in blob.tags if tag in tags_required])

    def get_word2vec_vector(self, sentence: str) -> np.ndarray:
        """
        Generates a Word2Vec vector for the input sentence.

        :param sentence: The input sentence for which to generate a vector.
        :return: The corresponding Word2Vec vector as a numpy array.
        """
        words = sentence.split()
        word_vectors = [
            self.word2vec_model.wv[word] for word in words if word in self.word2vec_model.wv
        ]
        return sum(word_vectors) / len(word_vectors) if word_vectors else np.zeros(self.word2vec_model.vector_size)

    def process_text(self, text: str) -> np.ndarray:
        """
        Processes the input text through cleaning, lemmatization, POS extraction, and vectorization.

        :param text: The input text to process.
        :return: The Word2Vec vector representing the processed text.
        """
        cleaned_text = self.clean_text(text)
        lemmatized_text = self.lemmatize_text(cleaned_text)
        pos_tags_text = self.get_POS_tags(lemmatized_text)
        return self.get_word2vec_vector(pos_tags_text)

class Predictor:
    def __init__(self, classification_model: any, text_processor: TextProcessor) -> None:
        """
        Initializes the Predictor with a classification model and a text processor.

        :param classification_model: The model used for classification.
        :param text_processor: The TextProcessor instance for processing text.
        """
        self.classification_model = classification_model
        self.text_processor = text_processor

    def predict(self, text: str) -> str:
        """
        Predicts the class of the input text using the classification model.

        :param text: The input text to classify.
        :return: The predicted class label.
        """
        word_vector = self.text_processor.process_text(text)
        prediction_index = self.classification_model.predict([word_vector])[0]
        prediction_label = classification_types.get(prediction_index, "Unknown")
        return prediction_label

@app.route('/', methods=['GET', 'POST'])
def index() -> str:
    """
    Renders the main page and handles form submissions for text classification.

    :return: The rendered HTML page.
    """
    prediction: Optional[str] = None
    if request.method == 'POST':
        text = request.form['text']
        processor = TextProcessor(word2vec_model)
        predictor = Predictor(support_classification_model, processor)
        prediction = predictor.predict(text)
        print("Ticket classified as:", prediction)  # Essential debug statement

    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
