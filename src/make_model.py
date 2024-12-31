from src.helpers.helper import word_to_features, parse_pos_file_new
from src.use_model import *
from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from dotenv import load_dotenv

def extract_features(sentences, tags):
  X_train_features = []
  y_train_flat = []

  for sentence, sentence_tags in zip(sentences, tags):
    for i in range(len(sentence)):
      features = word_to_features(sentence, i)  
      X_train_features.append(features)  
      y_train_flat.append(sentence_tags[i])

  return X_train_features, y_train_flat

def vectorize_data(X_train_features):
  vectorizer = DictVectorizer(sparse=False)
  X_train_vectorized = vectorizer.fit_transform(X_train_features)
  save_vectorizer(vectorizer)

  return X_train_vectorized, vectorizer

def train_new_model(X_train_vectorized, y_train_flat):
  model = RandomForestClassifier(n_estimators=100, random_state=42, verbose=True, n_jobs=3)
  model.fit(X_train_vectorized, y_train_flat)
  save_model(model)

  return model

def evaluate_model(sentences, tags):
  model = load_model()
  vectorizer = load_vectorizer()

  X_test_features = [word_to_features(sentence, i) for sentence in sentences for i in range(len(sentence))]
  X_test_vectorized = vectorizer.transform(X_test_features)

  y_test_flat = [tag for tag_seq in tags for tag in tag_seq]
  y_pred = model.predict(X_test_vectorized)

  print(f"Accuracy: {(accuracy_score(y_test_flat, y_pred)*100):.4f}\n")
  print(f"Classification report: \n{classification_report(y_test_flat, y_pred)}")
  return True

def make():
  load_dotenv()
  sentences, tags = parse_pos_file_new(file_path=os.getenv('CLEANED_DATA_PATH'))
  X_train_features, y_train_flat = extract_features(sentences, tags)
  X_train_vectorized, vectorizer = vectorize_data(X_train_features)
  model = train_new_model(X_train_vectorized, y_train_flat)
  evaluate_model(sentences, tags)
