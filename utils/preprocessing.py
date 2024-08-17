import nltk
from nltk.stem import WordNetLemmatizer
import numpy as np

nltk.download('punkt')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    tokens = nltk.word_tokenize(text)
    tokens = [lemmatizer.lemmatize(token.lower()) for token in tokens]
    return np.array(tokens)

def preprocess_texts(intents):
    # Example of preprocessing all texts in your intents.json
    texts = []
    labels = []
    for intent in intents['intents']:
        for pattern in intent['patterns']:
            texts.append(preprocess_text(pattern))
            labels.append(intent['tag'])
    return np.array(texts), np.array(labels)

def encode_labels(labels):
    # Encode labels into a one-hot format or other numerical encoding
    ...
