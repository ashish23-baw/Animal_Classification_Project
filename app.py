from flask import Flask, render_template, request
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)

# 1. Model Load karein
MODEL_PATH = 'model/animal_model.h5'
model = tf.keras.models.load_model(MODEL_PATH)

# Image size (Ensure karein ye training size se match kare)
IMG_SIZE = (224, 224) 

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return "No image uploaded", 400
    
    img_file = request.files['image']
    if img_file.filename == '':
        return "No image selected", 400

    if img_file:
        # Image save karein
        upload_path = os.path.join('static', img_file.filename)
        img_file.save(upload_path)
        
        # Pre-processing
        test_image = image.load_img(upload_path, target_size=IMG_SIZE)
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)
        
        # Note: Agar result abhi bhi galat aaye toh is line ko '#' se band kar dena
        test_image = test_image / 255.0 

        # Prediction
        prediction = model.predict(test_image)
        result = float(prediction[0][0])
        print(f"DEBUG SCORE: {result}")

        # --- FINAL FLIPPED LOGIC ---
        # Agar Cow ko Buffalo bata raha tha, toh humne '<' ko '>' kar diya hai
        if result > 0.5:
            # Ab ye Buffalo ke liye hai
            predicted_class = "Buffalo"
            breed_name = "Murrah"        
            animal_color = "Jet Black"    
            conf = result * 100
        else:
            # Ab ye Cow ke liye hai
            predicted_class = "Cow"
            breed_name = "Holstein Friesian" 
            animal_color = "Black & White"   
            conf = (1 - result) * 100

        return render_template('index.html', 
                               animal=predicted_class, 
                               breed=breed_name, 
                               color=animal_color,
                               confidence=round(conf, 2))

if __name__ == '__main__':
    app.run(debug=True)