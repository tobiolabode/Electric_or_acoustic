# import the necessary packages
import tensorflow as tf
from tensorflow import keras
import io
import base64
import numpy as np
import imutils
import cv2
from flask import request
from flask import jsonify
from flask import Flask

app = Flask(__name__)


def get_model():
    global model
    model = keras.models.load_model("my_model.h5")
    print(" * Model loaded!")


def preprocess_image(image):
    # # load the image
    # image = cv2.imread()
    # orig = image.copy()

    # pre-process the image for classification
    image = cv2.resize(image, (28, 28))
    image = image.astype("float") / 255.0
    image = keras.preprocessing.image.img_to_array(image)
    image = np.expand_dims(image, axis=0)

    return image


print("loading model")
get_model()


@app.route("/predict", methods=["POST"])
def predict():
    message = request.get_json(force=True)
    encoded = message["image"]
    readencoded = cv2.imread(encoded)
    decoded = base64.b64decode(readencoded)
    image = cv2.imread(io.BytesIO(decoded))
    preprocessed_image = preprocess_image(image)

    prediction = model.predict(preprocessed_image).tolist()

    response = {
        "prediction": {
            "acoustic_guitar": prediction[0][0],
            "electric_guitar": prediction[0][1]
        }
    }

    return jsonify(response)
