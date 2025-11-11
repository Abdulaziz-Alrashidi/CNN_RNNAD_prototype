from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
from PIL import Image
import io
import base64

#Load the model
model = tf.keras.models.load_model('final_model.keras')
#Create an instance of Flask
app = Flask(__name__)
#It will route post method into the function predict


@app.route("/predict", methods=["POST"])
def predict():

    #Save the json body into a variable and get the image from the response
    data = request.json
    image_64 = data["image"]

    #Convert the json body that contains the image data into image byte then reconstruct it into an image
    image_byte = base64.b64decode(image_64)
    image = Image.open(io.BytesIO(image_byte)).convert("RGB")

    #Resize the image to be compatiable, convert it into an array, and add a batch dimention
    image = image.resize((64, 64))
    image_array = np.asarray(image)
    image_array = np.expand_dims(image_array, axis=0)

    #Use the array to make a prediction then convert it to a list, so that it can be jsonified
    prediction = model.predict(image_array).tolist()
    #Retun the jsonified prediction
    return jsonify({"prediction": prediction})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)