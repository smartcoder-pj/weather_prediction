from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load trained model
model = pickle.load(open("weather_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
encoder = pickle.load(open("encoder.pkl", "rb"))


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    temperature = float(request.form["temperature"])
    humidity = float(request.form["humidity"])
    windspeed = float(request.form["windspeed"])

    # Prepare input
    input_data = [[temperature, humidity, windspeed]]

    # Scale input
    input_scaled = scaler.transform(input_data)

    # Predict
    prediction = model.predict(input_scaled)

    # Convert prediction to weather name
    weather = encoder.inverse_transform(prediction)[0]

    return render_template(
        "index.html",
        prediction=weather
    )


if __name__ == "__main__":
    app.run(debug=True)
