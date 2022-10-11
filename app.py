from flask import Flask, render_template, request, redirect, url_for
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import model_from_json
abc = open("D:/Datasets/Stroke.json", "r")
loaded_data = abc.read()
model = model_from_json(loaded_data)

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        lst = [float(x) for x in request.form.values()]
        arr = [np.array([lst, ])]
        output = model.predict(arr)

    return render_template('results.html', prediction_text=" Stroke : {}".format(output))


if __name__ == "__main__":
    app.run(debug=True)
