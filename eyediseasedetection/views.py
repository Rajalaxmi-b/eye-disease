from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
import tensorflow as tf
import numpy as np
import cv2
import joblib
import os

# Load model ONCE (important for performance)
# BASE_DIR is the Eyedisease project dir, but DL is in parent dir
PROJECT_ROOT = os.path.join(os.getcwd(), '..')

MODEL_PATH = os.path.join(PROJECT_ROOT, 'DL', 'eye_disease_model (1).h5')

LABEL_ENCODER_PATH = os.path.join(PROJECT_ROOT, 'DL', 'label_to_int.pkl')

try:
    model = tf.keras.models.load_model(MODEL_PATH)
    label_to_int = joblib.load(LABEL_ENCODER_PATH)
    int_to_label = {v: k for k, v in label_to_int.items()}
except Exception as e:
    model = None
    label_to_int = None
    int_to_label = None
    print(f"Error loading model or label encoder: {e}")


def home(request):
    prediction = None
    confidence = None
    error = None
    image_url = None

    if model is None or label_to_int is None:
        error = "Model or label encoder could not be loaded. Please check the file paths."
        return render(request, 'eyediseasedetection/index.html', {
            'prediction': prediction,
            'confidence': confidence,
            'error': error,
            'image_url': image_url
        })

    if request.method == 'POST' and request.FILES.get('eye_image'):
        try:
            image_file = request.FILES['eye_image']

            # Save uploaded image
            fs = FileSystemStorage()
            filename = fs.save(image_file.name, image_file)
            image_path = fs.path(filename)
            image_url = fs.url(filename)  # Set URL immediately after saving

            # IMAGE PREPROCESSING (MATCH TRAINING)
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                error = "Could not read the image. Please ensure it's a valid image file."
            else:
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                img_clahe = clahe.apply(img)
                img_resized = cv2.resize(img_clahe, (100, 100))
                img_array = img_resized / 255.0
                img_array = np.expand_dims(img_array, axis=[0, -1])  # Add batch and channel dimensions

                # PREDICTION
                predictions = model.predict(img_array)
                predicted_class = np.argmax(predictions)
                confidence = round(np.max(predictions) * 100, 2)

                prediction = int_to_label[predicted_class]

        except Exception as e:
            error = f"An error occurred during processing: {str(e)}"

    return render(request, 'eyediseasedetection/index.html', {
        'prediction': prediction,
        'confidence': confidence,
        'error': error,
        'image_url': image_url
    })
