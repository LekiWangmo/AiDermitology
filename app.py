from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics.pairwise import cosine_similarity
import pickle
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the model and data
model = tf.keras.models.load_model('skincare_model.h5')
df = pd.read_csv('result.csv')

with open('model.pkl', 'rb') as f:
    one_hot_encodings = pickle.load(f)

features = ['normal', 'dry', 'oily', 'combination', 'acne', 'sensitive', 'fine_lines', 
            'wrinkles', 'redness', 'dull', 'pore', 'pigmentation', 'blackheads', 
            'whiteheads', 'blemishes', 'dark_circles', 'eye_bags', 'dark_spots']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/submit_quiz', methods=['POST'])
def submit_quiz():
    try:
        user_responses = request.json
        user_vector = np.array([int(user_responses.get(feature, 0)) for feature in features])
        
        # Compute cosine similarity scores
        cs_values = cosine_similarity(user_vector.reshape(1, -1), one_hot_encodings)
        df['cs'] = cs_values[0]
        recommendations = df.sort_values('cs', ascending=False).head(10)
        
        # Prepare recommendations to send back to the client
        recommended_products = recommendations[['brand', 'name', 'price', 'skin type', 'concern', 'image_url']].to_dict(orient='records')
        
        return jsonify(recommended_products)
    except Exception as e:
        app.logger.error(f'An error occurred: {e}')  # Log the error
        return jsonify({'error': 'An error occurred while processing your request.'}), 500

@app.route('/index')
def home_page():
    return render_template('index.html')

@app.route('/scan')
def scan():
    return render_template('scan.html')

@app.route('/product')
def product():
    return render_template('product.html')

if __name__ == '__main__':
    app.run()
