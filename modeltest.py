
import tensorflow as tf
import numpy as np


def model_prediction(test_image):
    model = tf.keras.models.load_model("fruit_veg_CNN.h5")
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(64,64))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  
    predictions = model.predict(input_arr)
    return np.argmax(predictions)  


result_index = model_prediction("71xkI-PIE5L.jpg")
        
        
with open("labels.txt") as f:
    content = f.readlines()
labels = [label.strip() for label in content]

print(f"Prediction: This is a {labels[result_index]}")
