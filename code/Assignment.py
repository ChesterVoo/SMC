import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
import nltk

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

# OPTIONAL: Function for spell checking (skip if it takes too long)
# If needed, uncomment and install a faster library like 'pyspellchecker'
def spell_check(text):
     from spellchecker import SpellChecker
     spell = SpellChecker()
     words = word_tokenize(text)
     corrected_words = [spell.correction(word) for word in words]
     return ' '.join(corrected_words)

# Preprocessing: Apply lemmatization (and optional spell-checking)
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