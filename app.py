from flask import Flask, request, jsonify
from utils.preprocessing import preprocess_text
import tensorflow as tf
import numpy as np

app = Flask(__name__)

model = tf.keras.models.load_model('chatbot_model.h5')
intents = ...  # Load your intents.json data here

@app.route('/chat', methods=['POST'])
def chat():
    message = request.json['message']
    processed_message = preprocess_text(message)
    # Predict the intent
    intent = model.predict(processed_message)
    # Generate a response based on the intent
    response = generate_response(intent)
    return jsonify({"response": response})

if __name__ == '__main__':
    app.run(debug=True)
