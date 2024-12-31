import re
from xml.etree import ElementTree as ET
from src.helpers.constants import punctuations, valid_contexts
import random

def cleaner(input_file_path, output_file_path):
  with open(input_file_path, 'r', encoding='utf-8') as file:
    content = file.read()

  new_content = re.sub(r'(\d)>', r'\1">', content)

  with open(output_file_path, 'w', encoding='utf-8') as file:
    file.write(new_content)

  print(f"Replacements completed. Updated file saved as: {output_file_path}")

def parse_pos_file_new(file_path):
  tree = ET.parse(file_path)
  root = tree.getroot()

  sentences = []
  tags = []

  for sentence in root.findall('Sentence'):
    words_tags = sentence.text.strip().split(' ')
    words = [wt.split('_')[0] for wt in words_tags]
    print(words)
    sentence_tags = [wt.split('_')[1] for wt in words_tags]
    sentences.append(words)
    tags.append(sentence_tags)

  return sentences, tags

def word_to_features(sentence, i):
  word = sentence[i]
  features = {
    'word.lower()': word.lower(),
    'word.isupper()': word.isupper(),
    'word.istitle()': word.istitle(),
    'word.isdigit()': word.isdigit(),
  }

  if i > 0:
    features.update({
      'prev_word.lower()': sentence[i - 1].lower(),
      'prev_word.istitle()': sentence[i - 1].istitle(),
      'prev_word.isupper()': sentence[i - 1].isupper(),
    })
  else:
    features['BOS'] = True

  if i < len(sentence) - 1:
    features.update({
      'next_word.lower()': sentence[i + 1].lower(),
      'next_word.istitle()': sentence[i + 1].istitle(),
      'next_word.isupper()': sentence[i + 1].isupper(),
    })
  else:
    features['EOS'] = True

  return features

def tokenize_sentence(sentence):
  # return sentence.split()
  return re.findall(r'\w+|[^\w\s]', sentence)

def create_context(word):
  context = f"{valid_contexts[random.randint(0, len(valid_contexts) - 1)]}{word} {punctuations[random.randint(0, len(punctuations) - 1)]}"

  return context