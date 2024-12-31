import os
import nltk
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

nltk_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'nltk_data')
nltk.data.path = [nltk_dir]
nltk.download('averaged_perceptron_tagger', download_dir=nltk_dir)

def analyze_sentiment(text):
  sid = SentimentIntensityAnalyzer()
  sentiment_scores = sid.polarity_scores(text)
  print(sentiment_scores)

  if sentiment_scores['compound'] >= 0.05:
    return "Positive"
  elif sentiment_scores['compound'] <= -0.05:
    return "Negative"
  else:
    return "Neutral"
