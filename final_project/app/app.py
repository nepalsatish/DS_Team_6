from flask import Flask, request, render_template
from joblib import load
import numpy as np


app = Flask(__name__)

# Load the model from the file
model = load("../inf_model.joblib")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["GET", "POST"])
def predict():
    int_features = [float(x) for x in request.form.values()]
    features = [np.array(int_features)]
    prediction = model.predict(features)

    output = round(prediction[0], 2)

    return render_template(
        "index.html", prediction_text="Predicted Inflation Rate is {}%".format(output)
    )


if __name__ == "__main__":
    app.run()
