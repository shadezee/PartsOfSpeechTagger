import pickle
import os
from src.helpers.helper import create_context, tokenize_sentence, word_to_features
from src.helpers.constants import punctuations, tag_replacements
from src.sentiment_analyzer import analyze_sentiment

def save_model(model):
  os.makedirs(os.path.dirname("./build/model.pkl"), exist_ok=True)
  with open('./build/model.pkl', 'wb') as f:
    pickle.dump(model, f)

  print("Model saved successfully!")

def load_model():
  with open('./build/model.pkl', 'rb') as f:
    model = pickle.load(f)

  return model

def save_vectorizer(vectorizer):
  os.makedirs(os.path.dirname("./build/v.pkl"), exist_ok=True)
  with open("./build/v.pkl", "wb") as vectorizer_file:
    pickle.dump(vectorizer, vectorizer_file)

  print("Vectorizer saved successfully!")

def load_vectorizer():
  with open("./build/v.pkl", "rb") as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

  return vectorizer

def validate_tags(predicted_tags, sentence, model, vectorizer):
  if predicted_tags[-1] == "PUNCT" and sentence[-1] not in punctuations:
    print(f"Adjusting: {sentence[-1]}")

    context_sentence = create_context(sentence[-1])
    context_features = [word_to_features(context_sentence, i) for i in range(len(context_sentence))]
    context_features_vectorized = vectorizer.transform(context_features)
    context_tags = model.predict(context_features_vectorized)

    predicted_tags[-1] = context_tags[context_sentence.index(sentence[-1])]
  return predicted_tags

def run(sentence):
  sentiment = analyze_sentiment(sentence)
  sentence = tokenize_sentence(sentence)
  model  = load_model()
  vectorizer = load_vectorizer()
  features = [word_to_features(sentence, i) for i in range(len(sentence))]
  features_vectorized = vectorizer.transform(features)
  predicted_tags = validate_tags(model.predict(features_vectorized), sentence, model, vectorizer)
  # predicted_tags = model.predict(features_vectorized)

  print(f"\nSentiment: {sentiment}\n")
  for word, tag in zip(sentence, predicted_tags):
    tag = tag_replacements.get(tag, tag)
    print(f"{word} --> {tag}")

  return predicted_tags
