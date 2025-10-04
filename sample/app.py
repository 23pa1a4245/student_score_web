from flask import Flask, request, jsonify, render_template, send_from_directory
import pandas as pd
import joblib
import os

app = Flask(__name__, template_folder='frontend', static_folder='frontend/static')

# Load trained model
model_path = os.path.join("models", "trained_model.pkl")
model = joblib.load(model_path)

# Home page (prediction page)
@app.route('/')
def home():
    return render_template('index.html')

# About page
@app.route('/about')
def about():
    return render_template('about.html')

# Contact page
@app.route('/contact')
def contact():
    return render_template('contact.html')

# Prediction API
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    input_df = pd.DataFrame([data])
    input_df.columns = model.feature_names_in_
    prediction = model.predict(input_df)[0]
    return jsonify({"PredictedOverall": prediction})

if __name__ == '__main__':
    app.run(debug=True)
