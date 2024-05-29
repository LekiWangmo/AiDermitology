from flask import Flask, logging, request, jsonify, render_template
import numpy as np
import pandas as pd
import cv2
from img2vec_pytorch import Img2Vec # type: ignore
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

acne_classes = ['Acne', 'Blackheads', 'Eyebags', 'Pimple', 'Fairskin', 'unrecognized']

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

with open('one_hot_encodings.pkl', 'rb') as f:
    one_hot_encodings = pickle.load(f)

features = ['normal', 'dry', 'oily', 'combination', 'acne', 'sensitive', 'fine_lines',
            'wrinkles', 'dull', 'pore', 'blackheads',
            'whiteheads', 'dark_circles', 'eye_bags', 'dark_spots']

@app.route('/submit_quiz', methods=['POST'])
def submit_quiz():
    try:
        app.logger.debug('Received quiz submission.')
        user_responses = request.json
        app.logger.debug(f'user_responses: {user_responses}')
        
        user_vector = np.array([int(user_responses.get(feature, 0)) for feature in features])
        app.logger.debug(f'user_vector: {user_vector}')

        # Predict cosine similarity scores for all products
        cs_values = cosine_similarity(user_vector.reshape(1, -1), one_hot_encodings)
        app.logger.debug(f'cs_values: {cs_values}')
        
        df['cs'] = cs_values[0]

        recommendations = df.sort_values('cs', ascending=False).head(10) 
        
        recommended_products = recommendations[['brand', 'name', 'price', 'skin type', 'concern', 'image_url']].to_dict(orient='records')
        
        return jsonify(recommended_products)
    except Exception as e:
        app.logger.error(f'An error occurred: {e}', exc_info=True)
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
    app.run(debug=True)

