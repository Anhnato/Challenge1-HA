from flask import Flask, render_template, request, jsonify
import base64
import tensorflow as tf
import cv2
import numpy as np

app = Flask(__name__)

# Load your pre-trained model
model = tf.keras.models.load_model('keras.h5')
model.make_predict_function()
with open('class_names.txt') as file:
    class_labels = file.read().splitlines()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recognize', methods = ['POST'])
def recognize():
    if request.method == 'POST':
        data = request.get_json()

        # Decode base64 image data
        image_data = base64.b64decode(data['image'].split(',')[1])
        
        with open("temp.jpg", "wb") as temp:
            temp.write(image_data)
            
        with open('class_names.txt') as f:
            class_name = f.readlines()
        classs = [c.replace('/n','').replace('','') for c in class_name]
            
        image = cv2.imread('temp.jpg')
        image = cv2.resize(image, (28,28), interpolation = cv2.INTER_AREA)
        image_grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Get the predicted category
        predicted_category = np.reshape(image_grey,(28,28,1))
        predicted_category = (255-predicted_category.astype('float')) /255
        predicted = np.argmax(model.predict(np.array([predicted_category])), axis=-1)
        
    return jsonify({
        'prediction': 'your_prediction_result'})
    
if __name__ == '__main__':
    app.run(debug=True)
