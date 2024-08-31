import tensorflow
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import cv2

def predict():
    try:
        img = image.load_img(io.BytesIO('download.jpg'), target_size=(128, 128))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Predict the mood
        predictions = model.predict(img_array)
        mood_index = np.argmax(predictions[0])
        mood = class_labels.get(mood_index, 'unknown')
        print({'mood': mood})
    except:
        print ("error")
