from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load your pickled linear regression model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        exp = float(request.form["experience"])
        prediction = model.predict(np.array([[exp]]))
        salary = round(prediction[0], 2)

        return render_template("index.html", salary=salary, exp=exp)

    except:
        return render_template("index.html", error="Please enter a valid number.")

if __name__ == "__main__":
    app.run(debug=True)
