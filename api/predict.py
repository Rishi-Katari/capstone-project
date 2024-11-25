from flask import Flask, request, jsonify
import os
import numpy as np
import cv2
from tensorflow.keras.models import load_model # type: ignore

# Initialize Flask app
app = Flask(__name__)

# Load the saved model
MODEL_PATH = r"model.h5"  # Replace with your model's actual path
model = load_model(MODEL_PATH)

# Define your classes
disease_classes = [
    'actinic keratosis',
    'basal cell carcinoma',
    'dermatofibroma',
    'melanoma',
    'nevus',
    'pigmented benign keratosis',
    'seborrheic keratosis',
    'squamous cell carcinoma',
    'vascular lesion'
]

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file uploaded", 400
    
    file = request.files['file']
    if file.filename == '':
        return "No file selected", 400
    
    # Save uploaded file temporarily
    filepath = os.path.join('uploads', file.filename)
    file.save(filepath)

    # Load and preprocess the image
    img = cv2.imread(filepath)
    img = cv2.resize(img, (150, 150))  # Resize to match model input
    img = img / 255.0  # Normalize pixel values
    img = np.expand_dims(img, axis=0)  # Expand dimensions to match model input

    # Make predictions
    predictions = model.predict(img)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class = disease_classes[predicted_class_index]
    confidence = np.max(predictions) * 100  # Convert to percentage

    result = {
        "predicted_class": predicted_class,
        "confidence": f"{confidence:.2f}%"
    }

    return jsonify(result)

# Ensure the app is run only if this is the main module
if __name__ == "__main__":
    app.run(debug=True)
