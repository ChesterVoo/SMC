from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd

# Load the dataset
df = pd.read_csv('https://raw.githubusercontent.com/ChesterVoo/SMC/refs/heads/main/dataset/airlineTweets.csv')

# Initialize the sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

# Initialize variables to accumulate the sentiment scores
negative_scores = []
neutral_scores = []
positive_scores = []
compound_scores = []

# Analyze sentiment for each tweet and collect the scores
for tweet in df['text']:
    vs = analyzer.polarity_scores(tweet)
    negative_scores.append(vs['neg'])
    neutral_scores.append(vs['neu'])
    positive_scores.append(vs['pos'])
    compound_scores.append(vs['compound'])

# Calculate average scores
average_negative = sum(negative_scores) / len(negative_scores)
average_neutral = sum(neutral_scores) / len(neutral_scores)
average_positive = sum(positive_scores) / len(positive_scores)
average_compound = sum(compound_scores) / len(compound_scores)

# Print the average sentiment scores
print(f'Average Negative Sentiment: {average_negative}')
print(f'Average Neutral Sentiment: {average_neutral}')
print(f'Average Positive Sentiment: {average_positive}')
print(f'Average Compound Sentiment: {average_compound}')