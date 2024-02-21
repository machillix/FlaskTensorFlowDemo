# demo.py

from keras.models import load_model
import re
from io import BytesIO
import base64
from PIL import Image, ImageOps 
import numpy as np
from flask import Flask, jsonify, request

app = Flask(__name__)

@app.route("/", methods = ['POST'])
def ImageClassification():

    if request.method != 'POST':
        return "failed"
    
    jsonData = request.json

    np.set_printoptions(suppress=True)

    model = load_model("./keras_Model.h5", compile=False)
    classNames = open("./labels.txt", "r").readlines()

    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

    size = (224, 224)

    imageData = re.sub('^data:image/.+;base64,', '', jsonData.get('img'))
    image = Image.open(BytesIO(base64.b64decode(imageData)))
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

    data[0] = normalized_image_array

    prediction = model.predict(data)
    index = np.argmax(prediction)
    className = classNames[index]
    confidenceScore = prediction[0][index]

    print("Class:", className[2:], end="")
    print("Confidence Score:", confidenceScore)
    return jsonify(name=className[2:], conf=str(confidenceScore))
