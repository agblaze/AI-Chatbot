import json
import numpy as np
import tensorflow as tf
from utils.preprocessing import preprocess_texts, encode_labels

with open('intents.json') as f:
    intents = json.load(f)

# Preprocess the data
texts, labels = preprocess_texts(intents)
encoded_labels = encode_labels(labels)

# Define the model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, input_shape=(len(texts[0]),), activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(len(set(labels)), activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(texts, encoded_labels, epochs=200, batch_size=8)

model.save('chatbot_model.h5')
