# Sentiment Analysis Tool
# Analyzes text sentiment using NLTK's VADER sentiment analyzer

import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Download VADER lexicon
nltk.download('vader_lexicon')

# Initialize analyzer
sid = SentimentIntensityAnalyzer()

# Function to analyze sentiment
def analyze_sentiment(text):
    scores = sid.polarity_scores(text)
    compound = scores['compound']
    if compound >= 0.05:
        return 'Positive'
    elif compound <= -0.05:
        return 'Negative'
    else:
        return 'Neutral'

# Example usage
texts = [
    "I love this product, it's amazing!",
    "This is the worst experience ever.",
    "The service was okay, nothing special."
]

for text in texts:
    sentiment = analyze_sentiment(text)
    print(f"Text: {text}\nSentiment: {sentiment}\n")