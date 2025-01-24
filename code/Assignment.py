import pandas as pd
import torch
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import DistilBertTokenizer, DistilBertModel
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score


# nltk.download('punkt')
# nltk.download('wordnet')
# nltk.download('punkt_tab')


# Load the dataset
data = pd.read_csv('https://raw.githubusercontent.com/ChesterVoo/SMC/refs/heads/main/dataset/airlineTweets.csv')

# Initialize the lemmatizer
lemmatizer = WordNetLemmatizer()

# Function for lemmatization
def lemmatize_text(text):
    words = word_tokenize(text)  # Tokenize the text into words
    lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(lemmatized_words)

# Function for spellchecker
def spell_check(text):
     from spellchecker import SpellChecker
     spell = SpellChecker()
     words = word_tokenize(text)
     corrected_words = [spell.correction(word) for word in words]
     return ' '.join(corrected_words)

# Preprocessing: Apply lemmatization 
data['cleaned_text'] = data['text'].apply(lambda x: lemmatize_text(x))

# Initialize VADER sentiment analyzer
sia = SentimentIntensityAnalyzer()

# Apply VADER sentiment analysis on the cleaned "text" column
data['sentiment'] = data['cleaned_text'].apply(lambda x: sia.polarity_scores(x))

# Extract the compound score and other sentiment scores
data['compound'] = data['sentiment'].apply(lambda x: x['compound'])
data['positive'] = data['sentiment'].apply(lambda x: x['pos'])
data['neutral'] = data['sentiment'].apply(lambda x: x['neu'])
data['negative'] = data['sentiment'].apply(lambda x: x['neg'])

# Calculate the average sentiments
average_neutral = data['neutral'].mean()
average_positive = data['positive'].mean()
average_negative = data['negative'].mean()

# Print the averages
print(f"Average Neutral Sentiment: {average_neutral:.4f}")
print(f"Average Positive Sentiment: {average_positive:.4f}")
print(f"Average Negative Sentiment: {average_negative:.4f}")

# Initialize DistilBERT
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = DistilBertModel.from_pretrained("distilbert-base-uncased")

# Function to compute DistilBERT embeddings
def compute_distilbert_embeddings(text):
    encoded_input = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
    with torch.no_grad():  # Disable gradient calculations for inference
        model_output = model(**encoded_input)
    # Extract the last hidden state (embedding) and take the mean across the sequence
    embeddings = model_output.last_hidden_state.mean(dim=1).squeeze().numpy()
    return embeddings

# Apply DistilBERT to the cleaned text
data['distilbert_embeddings'] = data['cleaned_text'].apply(lambda x: compute_distilbert_embeddings(x))

# Display the updated dataframe
print(data.head())

# Logistic Regression Classifier
# Create sentiment labels based on compound scores
def create_sentiment_label(compound_score):
    if compound_score > 0:
        return 'positive'
    elif compound_score < 0:
        return 'negative'
    else:
        return 'neutral'

data['sentiment_label'] = data['compound'].apply(create_sentiment_label)

# Convert DistilBERT embeddings into a NumPy array
embedding_matrix = np.array(data['distilbert_embeddings'].tolist())

# Prepare features (X) and labels (y)
X = embedding_matrix
y = data['sentiment_label']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Define the parameter grid for Logistic Regression
param_grid = {
    'C': [0.01, 0.1, 1, 10],  # Different inverse regularization strengths
    'penalty': ['l2'],        # Use only L2 regularization for simplicity
    'solver': ['liblinear']  # Solvers for Logistic Regression
}

# Initialize Grid Search
grid_search = GridSearchCV(
    estimator=LogisticRegression(max_iter=1000),
    param_grid=param_grid,
    cv=3,
    scoring='accuracy',
    verbose=1,
    n_jobs=-1
)

# Fit the model with Grid Search
grid_search.fit(X_train, y_train)

# Get the best model and its parameters
best_model_lr = grid_search.best_estimator_
best_params_lr = grid_search.best_params_
print(f"Best parameters: {best_params_lr}")

# Evaluate the best model on the test set
y_pred_lr = best_model_lr.predict(X_test)

# Display the classification report and accuracy
print("Classification Report:")
print(classification_report(y_test, y_pred_lr))
print(f"Accuracy: {accuracy_score(y_test, y_pred_lr):.4f}")


# SVM 
# Create sentiment labels based on compound scores
def create_sentiment_label(compound_score):
    if compound_score > 0:
        return 'positive'
    elif compound_score < 0:
        return 'negative'
    else:
        return 'neutral'

# Add sentiment labels to the dataset
data['sentiment_label'] = data['compound'].apply(create_sentiment_label)

# Convert DistilBERT embeddings into a NumPy array
embedding_matrix = np.array(data['distilbert_embeddings'].tolist())

# Prepare features and labels
X = embedding_matrix  # DistilBERT embeddings
y = data['sentiment_label']  # Sentiment labels

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Define the parameter grid for SVM
param_grid = {
    'C': [0.01, 0.1, 1, 10],      # Regularization parameter
    'kernel': ['linear', 'rbf'], # Kernel type: linear or radial basis function
    'gamma': ['scale', 'auto']   # Kernel coefficient
}

# Initialize the SVM classifier
svm_classifier = SVC()

# Perform Grid Search with cross-validation
grid_search = GridSearchCV(
    estimator=svm_classifier, param_grid=param_grid, cv=3,
    scoring='accuracy', verbose=1, n_jobs=-1
)
grid_search.fit(X_train, y_train)

# Get the best SVM model and parameters
best_model = grid_search.best_estimator_
best_params = grid_search.best_params_
print(f"Best Parameters: {best_params}")

# Evaluate the best model on the test set
y_pred = best_model.predict(X_test)

# Display classification results
print("Classification Report:")
print(classification_report(y_test, y_pred))
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")