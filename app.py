from flask import Flask, logging, request, jsonify, render_template
import numpy as np
import pandas as pd
import cv2
from img2vec_pytorch import Img2Vec
import pickle
from flask_cors import CORS
from PIL import Image
import base64
from io import BytesIO

import tensorflow as tf
from torch import cosine_similarity
app = Flask(__name__)
CORS(app)

# Load SVM model
with open("svm_model2.pkl", "rb") as f:
    svm_model = pickle.load(f)

img2vec = Img2Vec(model='resnet-18')

acne_classes = ['Acne', 'Blackheads', 'Eyebags', 'Pimple', 'Fairskin', 'unrecornized']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/classify', methods=['POST'])
def classify():
    uploaded_file = request.files['file']
    if uploaded_file:
        # Read the image
        image = Image.open(uploaded_file)
        # Convert to RGB
        image = image.convert("RGB")
        # Resize the image
        image = image.resize((264, 264))
        # Extract features using Img2Vec
        features = img2vec.get_vec(image).reshape(1, -1)
        # Predict the class label using SVM model
        predicted_class_index_svm = svm_model.predict(features)[0]
        predicted_class_svm = acne_classes[predicted_class_index_svm]
        # Convert image to base64 string
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        # Pass the image data and prediction results to the result page
        return render_template('result.html', predicted_class_svm=predicted_class_svm, image_data=image_base64)
    else:
        return render_template('upload23.html', error="No file selected.")

@app.route('/classify_camera', methods=['POST'])
def classify_camera():
    image_data = request.form['image_data']
    if image_data:
        # Decode the image data
        image_data = image_data.split(",")[1]
        image = Image.open(BytesIO(base64.b64decode(image_data)))
        # Convert to RGB
        image = image.convert("RGB")
        # Resize the image
        image = image.resize((264, 264))
        # Extract features using Img2Vec
        features = img2vec.get_vec(image).reshape(1, -1)
        # Predict the class label using SVM model
        predicted_class_index_svm = svm_model.predict(features)[0]
        predicted_class_svm = acne_classes[predicted_class_index_svm]
        # Convert image to base64 string
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        # Pass the image data and prediction results to the result page
        return render_template('result.html', predicted_class_svm=predicted_class_svm, image_data=image_base64)
    else:
        return render_template('scan.html', error="No file selected.")
            


# Load the model and data for recommendation
model = tf.keras.models.load_model('skincare_model.h5')
df = pd.read_csv('result.csv')


with open('model.pkl', 'rb') as f:
    one_hot_encodings = pickle.load(f)

features = ['normal', 'dry', 'oily', 'combination', 'acne', 'sensitive',
            'wrinkles', 'dull','blackheads',
             'dark_circles', 'eye_bags', 'dark_spots']
import random

@app.route('/submit_quiz', methods=['POST'])
def submit_quiz():
    try:
        user_responses = request.json
        user_vector = np.array([int(user_responses.get(feature, 0)) for feature in features])
        
        # Convert user_vector to a Tensor
        user_tensor = tf.convert_to_tensor(user_vector.reshape(1, -1), dtype=tf.float32)
        
        # Compute cosine similarity scores
        cs_values = tf.keras.losses.cosine_similarity(user_tensor, one_hot_encodings)
        df['cs'] = cs_values.numpy()[0]
        recommendations = df.sort_values('cs', ascending=False)
        
        # Randomly select 5 products from recommendations
        recommended_products = recommendations.sample(n=5)
        
        # Prepare recommendations to send back to the client
        recommended_products_data = recommended_products[['brand', 'name', 'price', 'skin type', 'concern', 'image_url']].to_dict(orient='records')
        
        return jsonify(recommended_products_data)
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

