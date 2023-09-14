import tensorflow as tf
from utils import preprocess_image
import numpy as np
from scipy import ndimage

def load_model():
    # Load your pre-trained ML model
    model = tf.keras.models.load_model('vgg19model.h5', compile = False)
    return model

def predict_image(image_path):
    model = load_model()
    img = preprocess_image(image_path)  # Implement this function to preprocess the image
    img = img.reshape(1, 224, 224, 3)
    prediction = model.predict(img)
    # Process the prediction and return a classification result as a string
    class_labels = [' Aloevera' , 'Amla' , 'Amruthballi', 'Arali', 'ashoka', ' Astma_weed', 'Badipala', 'Balloon_Vine',  'Bamboo', 'Beans', 'Betal', ' Bhrami', 'Bringaraja', 'camphor', 'Caricature', 'Castor', 'Catharanthus', 'Chakte', 'Chilly', 'Citron lime (herelikai)', 'Coffee', 'Common rue(nagadalli)', 'Coriender', 'Curry', 'Doddpathre', 'Drumstick', 'Ekka', 'Eucalyptus', 'Ganigale', 'Ganike', 'Gasagase', 'Ginger', 'Globe Amarnath', 'Guava', 'Henna', 'Hibiscus', 'Honge', 'Insulin', 'Jackfruit', 'Jasmine', 'Kamakasturi', 'Kambajala', 'Kasambruga', 'Kepala', 'Kohlrabi', 'Lantana', 'Lemon', 'Lemongrass', 'Malabar_Nut', 'Malabar_Spinach', 'Mango', 'Marigold', 'Mint', 'Neem', 'Nelavembu', 'Nerale', 'Nooni', 'Onion', 'Padri', 'Palak(Spinach)', 'Papaya', 'Parijatha', 'Pea', 'Pepper', 'Pomoegranate', 'Pumpkin', 'Raddish', 'Rose', 'Sampige', 'Sapota', 'Seethaashoka', 'Seethapala', 'Spinach1', 'Tamarind', 'Taro', 'Tecoma', 'Thumbe', 'Tomato', 'Tulsi', 'Turmeric']
    predicted_class_index = np.argmax(prediction) + 1
    result = class_labels[predicted_class_index]  # Replace with your actual result
    return result
