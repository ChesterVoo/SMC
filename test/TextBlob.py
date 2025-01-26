from textblob import TextBlob
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('https://raw.githubusercontent.com/ChesterVoo/SMC/refs/heads/main/dataset/airlineTweets.csv')

# Compute sentiment polarity and subjectivity using TextBlob
df['polarity'] = df['text'].apply(lambda x: TextBlob(str(x)).sentiment.polarity)
df['subjectivity'] = df['text'].apply(lambda x: TextBlob(str(x)).sentiment.subjectivity)

# Calculate the average polarity and subjectivity
average_polarity = df['polarity'].mean()
average_subjectivity = df['subjectivity'].mean()

# Display the polarity and subjectivity values
print("Polarity and Subjectivity values:")
print(df[['text', 'polarity', 'subjectivity']].sum)

# Print the averages
print(f"Average Polarity: {average_polarity}")
print(f"Average Subjectivity: {average_subjectivity}")

# Set up the figure for both plots
plt.figure(figsize=(12, 12))

# Histogram for polarity
plt.subplot(2, 1, 1)
plt.hist(df['polarity'], bins=30, alpha=0.7, color='skyblue', edgecolor='black', label='Polarity')
plt.title('Distribution of Sentiment Polarity (Using TextBlob)', fontsize=16)
plt.xlabel('Polarity', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.legend(fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Histogram for subjectivity
plt.subplot(2, 1, 2)
plt.hist(df['subjectivity'], bins=30, alpha=0.7, color='lightgreen', edgecolor='black', label='Subjectivity')
plt.title('Distribution of Sentiment Subjectivity (Using TextBlob)', fontsize=16)
plt.xlabel('Subjectivity', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.legend(fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Show the plots
plt.tight_layout()
plt.show()