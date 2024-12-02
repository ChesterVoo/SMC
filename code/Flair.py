import pandas as pd
from flair.models import TextClassifier
from flair.data import Sentence

# Load the dataset
df = pd.read_csv('https://raw.githubusercontent.com/ChesterVoo/SMC/refs/heads/main/dataset/airlineTweets.csv')

# Load the pre-trained sentiment analysis model
classifier = TextClassifier.load('en-sentiment')  # This loads a pre-trained English sentiment model

# Initialize lists to store sentiment scores
positive_scores = []
negative_scores = []

# Analyze sentiment for each tweet
for tweet in df['text']:
    # Convert tweet to a Flair Sentence object
    sentence = Sentence(tweet)
    
    # Predict sentiment
    classifier.predict(sentence)
    
    # Extract sentiment score (Flair model outputs positive and negative scores)
    positive_scores.append(sentence.labels[0].score if sentence.labels[0].value == 'POSITIVE' else 0)
    negative_scores.append(sentence.labels[0].score if sentence.labels[0].value == 'NEGATIVE' else 0)

# Calculate average sentiment scores
average_positive = sum(positive_scores) / len(positive_scores)
average_negative = sum(negative_scores) / len(negative_scores)

# Print the average sentiment scores
print(f'Average Positive Sentiment: {average_positive}')
print(f'Average Negative Sentiment: {average_negative}')