import numpy as np
import pandas as pd
import tensorflow as tf
import pickle

# Assuming your data and model setup
df = pd.read_csv('result.csv')
features = ['normal', 'dry', 'oily', 'combination', 'acne', 'sensitive', 'fine_lines', 
            'wrinkles', 'redness', 'dull', 'pore', 'pigmentation', 'blackheads', 
            'whiteheads', 'blemishes', 'dark_circles', 'eye_bags', 'dark_spots']

one_hot_encodings = np.zeros((len(df), len(features)))
for i in range(len(df)):
    for j, feat in enumerate(features):
        concern_value = df.iloc[i]['concern']
        if isinstance(concern_value, str) and feat in concern_value:
            one_hot_encodings[i][j] = 1
    if df.iloc[i]['skin type'] == 'all':
        one_hot_encodings[i][:4] = 1

# Define and train your model (example code)
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(len(features),)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1, activation='linear')
])
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

# Dummy training data for the example
user_df = pd.DataFrame([np.random.randint(2, size=len(features)) for _ in range(10)], columns=features)
custom_recommendations = np.random.rand(10, 1)

model.fit(user_df, custom_recommendations, epochs=10)

# Save the model
model.save('skincare_model.h5')

# Save the one_hot_encodings using pickle
with open('model.pkl', 'wb') as f:
    pickle.dump(one_hot_encodings, f)
