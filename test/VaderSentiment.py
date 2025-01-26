import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Load the data
data = pd.read_csv('https://raw.githubusercontent.com/ChesterVoo/SMC/refs/heads/main/dataset/airlineTweets.csv')

# Initialize VADER sentiment analyzer
sia = SentimentIntensityAnalyzer()

# Apply VADER sentiment analysis on the "text" column
data['sentiment'] = data['text'].apply(lambda x: sia.polarity_scores(x))

# Extract the compound score
data['compound'] = data['sentiment'].apply(lambda x: x['compound'])

# Calculate the average neutral, positive, and negative sentiments
average_neutral = data['sentiment'].apply(lambda x: x['neu']).mean()
average_positive = data['sentiment'].apply(lambda x: x['pos']).mean()
average_negative = data['sentiment'].apply(lambda x: x['neg']).mean()

# Print the averages
print(f"Average Neutral Sentiment: {average_neutral:.4f}")
print(f"Average Positive Sentiment: {average_positive:.4f}")
print(f"Average Negative Sentiment: {average_negative:.4f}")