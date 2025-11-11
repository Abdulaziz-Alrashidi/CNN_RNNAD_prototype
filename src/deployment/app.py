import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import json
from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
from PIL import Image
import io
import base64

#Load the model
model = tf.keras.models.load_model('final_model.keras')

#Extract the classes names
with open("class_names.json", "r") as f:
    classes = json.load(f)

#Create an instance of Flask
app = Flask(__name__)
#It will route post method into the function predict
@app.route("/predict", methods=["POST"])
def predict():
    try:

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

        #Make the prediction and map the class name and return the prediction and its class
        pred = model.predict(image_array)[0]
        pred_index = int(np.argmax(pred))
        pred_class = classes[pred_index]

        return jsonify({
                "prediction": pred.tolist(),
                "predicted_class": pred_class
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)