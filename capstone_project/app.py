from flask import Flask, request, render_template, jsonify
from tensorflow.keras.models import load_model  # type: ignore
import numpy as np
import cv2
import os

app = Flask(__name__)

# Load the saved model
MODEL_PATH = r"D:\capstone project\capstone project\cnn_modelskin.h5"  # Update the path to your saved model
model = load_model(MODEL_PATH)

# Define the disease classes
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

# Function to classify a single image
def classify_image(image_path):
    # Load and preprocess the image
    img = cv2.imread(image_path)

    if img is None:
        print(f"Error: Image not found at {image_path}")
        return None, None

    img = cv2.resize(img, (150, 150))  # Resize to match model input
    img = img / 255.0  # Normalize pixel values
    img = np.expand_dims(img, axis=0)  # Expand dimensions to match model input

    # Make predictions
    predictions = model.predict(img)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class = disease_classes[predicted_class_index]

    # Confidence level of prediction
    confidence = np.max(predictions) * 100  # Convert to percentage

    return predicted_class, confidence

# Route for the home page
@app.route('/')
def index():
    return render_template('index.html')

# Route for handling image upload and prediction
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

    try:
        # Use the classify_image function
        predicted_disease, confidence = classify_image(filepath)
        
        if predicted_disease is None:
            return f"Error in classifying the image: Image not found or corrupted", 500
        
        result = {
            "predicted_class": predicted_disease,
            "confidence": f"{confidence:.2f}%"
        }
    except Exception as e:
        return f"Error during prediction: {str(e)}", 500
    finally:
        # Remove temporary file
        os.remove(filepath)

    return jsonify(result)

if __name__ == "__main__":
    # Ensure the uploads directory exists
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    app.run(debug=True)
