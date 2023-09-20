import tensorflow as tf
from utils import preprocess_image
import numpy as np
from scipy import ndimage


def load_model():
    # Load your pre-trained ML model
    model = tf.keras.models.load_model('plant_identification_model2.h5', compile=False)
    return model


def predict_image(image_path):
    model = load_model()
    img = preprocess_image(image_path)  # Implement this function to preprocess the image
    img = img.reshape(1, 224, 224, 3)
    predicted = model.predict(img)
    prediction = np.argmax(predicted)
    # Process the prediction and return a classification result as a string
    class_labels = ['Alpinia Galanga (Rasna)', 'Amaranthus Virdis (Arive-Dantu)', 'Artocarpus Heterophyllus (Jackfruit)', 'Azadirachta Indica (Neem)', 'Basella Alba (Basale)', 'Brassica Juncea (Indian Mustard)', 'Carissa Carandas (Karanda)', 'Citrus Limon (Lemon)', 'Ficus Auriculata (Roxburgh fig)', 'Ficus Religiosa (Peepal Tree)', 'Hibiscus Rosa-sinensis', 'Jasminum (Jasmine)', 'Mangifera Indica (Mango)', 'Mentha (Mint)', 'Moringa Oleifera (Drumstick)', 'Muntingia Calabura (Jamaica Cherry-Gasagase)', 'Murraya Koenigii (Curry)', 'Nerium Oleander (Oleander)', 'Nyctanthes Arbor-tristis (Parijata)', 'Ocimum Tenuiflorum (Tulsi)', 'Piper Betle (Betel)', 'Plectranthus Amboinicus (Mexican Mint)', 'Pongamia Pinnata (Indian Beech)', 'Psidium Guajava (Guava)', 'Punica Granatum (Pomegranate)', 'Santalum Album (Sandalwood)', 'Syzygium Cumini (Jamun)', 'Syzygium Jambos (Rose Apple)', 'Tabernaemontana Divaricata (Crape Jasmine)', 'Trigonella Foenum-graecum (Fenugreek)']
    # predicted_class_index = np.argmax(prediction)
    result = class_labels[prediction]  # Replace with your actual result
    # result = np.array2string(predicted, precision=2, separator=' ', suppress_small=True)
    return result
