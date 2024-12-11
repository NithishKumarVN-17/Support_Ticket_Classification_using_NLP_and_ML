# Support Ticket Classification using NLP and Machine Learning

## Project Overview
This project develops a machine learning solution for classifying support tickets using Natural Language Processing (NLP) techniques. The goal is to automatically categorize support tickets from JP Morgan and Chase Bank's dataset into predefined classes.

## Dataset
- Source: Kaggle - JP Morgan and Chase Bank Support Ticket Dataset
- Characteristics: Text-based support tickets with confidential data masked

## Data Preprocessing
### Text Cleaning Techniques
- Stopwords removal
- Punctuation elimination
- Part-of-Speech (POS) tag filtering

## Exploratory Data Analysis
### Text Analysis Techniques
- Bigram and trigram analysis
- Word cloud generation
- Topic modeling using Latent Dirichlet Allocation (LDA)
  - Extracted 5 distinct topic classifications

## Feature Engineering
### Text Vectorization Methods
- Count Vectorizer
- TF-IDF Vectorizer
- Word2Vec Vectorizer

## Machine Learning Models
### Implemented Models
- Logistic Regression
- Random Forest
- (Other models as applicable)

### Model Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1-Score

### Hyperparameter Tuning
- Conducted on the best-performing model

## Web Application
### Technologies Used
- Flask
- Machine Learning Model Deployment

### Application Features
- User input interface
- Real-time ticket classification
- Results display in UI

## Setup and Installation

### Prerequisites
- Python 3.8+
- Required libraries (see `requirements.txt`)

### Installation Steps
```bash
# Clone the repository
git clone https://github.com/yourusername/support-ticket-classification.git

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

# Install dependencies
pip install -r requirements.txt

# Run the application
python app.py
```


## Future Improvements
- Implement more advanced NLP techniques
- Expand classification categories
- Improve model accuracy
- Add more robust error handling
