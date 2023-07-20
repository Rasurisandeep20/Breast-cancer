from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from keras.models import load_model
from tensorflow.keras.utils import load_img, img_to_array
import cv2

app = Flask(__name__)

dic = {0: 'Breast Cancer Not Detected', 1: 'Breast Cancer Detected'}

# Load the trained model
img_size = 256
model = load_model('cnn_model.h5')
# model = load_model('weights.best.hdf5')
model.make_predict_function()

def predict_label(img_path):
    # Load the image, convert it to grayscale and resize it
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  
    resized = cv2.resize(gray, (img_size, img_size))
    
    # Normalize the image pixel values and reshape it to match the input shape of the model
    i = img_to_array(resized) / 255.0
    i = i.reshape(1, img_size, img_size, 1)
    
    # Make a prediction on the input image and return the predicted label
    predict_x = model.predict(i)
    p = np.argmax(predict_x, axis=1)
    return dic[p[0]]

@app.route("/", methods=["GET", "POST"])
def homepage():
    return render_template('upload.html')

@app.route("/upload", methods=["GET", "POST"])
def upload():
    description = None
    img_path = None
    
    if request.method == "POST" and 'photo' in request.files:
        # Save the uploaded image to a file
        img = request.files['photo']
        img_path = 'static/img/' + img.filename
        img.save(img_path)
        
        # Make a prediction on the uploaded image
        description = predict_label(img_path)
    
    return render_template('upload.html', cp=description, src=img_path)

if __name__ == "__main__":
    app.run(debug=True)
