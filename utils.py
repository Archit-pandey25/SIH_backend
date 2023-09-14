import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image

def preprocess_image(image_path) :
    datagen = ImageDataGenerator(rescale=1./255,
                             shear_range=0.2,
                             zoom_range=0.2,
                             horizontal_flip=True)
#    img_path = r'C:\Users\tiger\Pictures\1542.jpg'
    img = image.load_img(image_path, target_size=(224,224))  # Replace IMAGE_SIZE with your model's input size
#    img = img.reshape(1, 224, 224, 3)
    img = image.img_to_array(img)
    img = datagen.random_transform(img)
    img = datagen.standardize(img)
    return img