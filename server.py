from flask import Flask, jsonify, request 
from flask_cors import CORS
  

app = Flask(__name__) 
CORS(app)  
  
@app.route('/', methods = ['GET', 'POST']) 
def home():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            import tensorflow as tf
            import numpy as np
            from os import remove

            def model_prediction(test_image):
                model = tf.keras.models.load_model("fruit_veg_CNN.h5")
                image = tf.keras.preprocessing.image.load_img(test_image, target_size=(64,64))
                input_arr = tf.keras.preprocessing.image.img_to_array(image)
                input_arr = np.array([input_arr])  
                predictions = model.predict(input_arr)
                return np.argmax(predictions)  

            file.save(file.filename)
            result_index = model_prediction(file.filename)
                    
            with open("labels.txt") as f:
                content = f.readlines()
            labels = [label.strip() for label in content]

            print(f"Prediction: This is a {labels[result_index]}")
            remove(file.filename)
            return labels[result_index]
            return jsonify({'data': labels[result_index]})
            


    data = "no-file"
    return data
    return jsonify({'data': data})
  

if __name__ == '__main__': 
  
    app.run(host="0.0.0.0",debug = False) 
