import warnings
import logging
import numpy as np
import pandas as pd
import joblib
from flask import Flask, render_template, request

warnings.filterwarnings('ignore')

app = Flask(__name__, template_folder='C:/Users/harih/OneDrive/Documents/hackathon/templates')

# Set up logging
logging.basicConfig(level=logging.DEBUG)

# Load data and models
try:
    data = pd.read_csv('merged.csv')
    rf_model = joblib.load('rf_model.pkl')
    scaler = joblib.load('scaler.pkl')
    app.logger.info("Models and data loaded successfully.")
except Exception as e:
    app.logger.error(f"Error loading models or data: {e}")

# Crop dictionary
crop_dict = {
    'cotton': 0, 'horsegram': 1, 'jowar': 2, 'maize': 3, 'moong': 4,
    'ragi': 5, 'rice': 6, 'sunflower': 7, 'wheat': 8, 'sesamum': 9,
    'soyabean': 10, 'rapeseed': 11, 'jute': 12, 'arecanut': 13, 'onion': 14,
    'potato': 15, 'sweetpotato': 16, 'tapioca': 17, 'turmeric': 18, 'barley': 19,
    'banana': 20, 'coriander': 21, 'garlic': 22, 'blackpepper': 23, 'cardamom': 24,
    'cashewnuts': 25, 'blackgram': 26, 'coffee': 27, 'ladyfinger': 28, 'brinjal': 29,
    'cucumber': 30, 'grapes': 31, 'mango': 32, 'orange': 33, 'papaya': 34,
    'tomato': 35, 'cabbage': 36, 'bottlegourd': 37, 'pineapple': 38, 'carrot': 39,
    'radish': 40, 'bittergourd': 41, 'drumstick': 42, 'jackfruit': 43,
    'cauliflower': 44, 'watermelon': 45, 'ashgourd': 46, 'beetroot': 47,
    'pomegranate': 48, 'ridgegourd': 49, 'pumpkin': 50, 'apple': 51, 'ginger': 52
}

# Season index mapping
season_index = {
    'Kharif': 0,
    'Rabi': 1,
    'Summer': 2,
    'Whole Year': 3
}

@app.route('/')
def home():
    return render_template('index(4).html', prediction1=None, prediction2=None, prediction3=None)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        app.logger.info("Received request for prediction.")
        N = float(request.form['nitrogen'])
        P = float(request.form['phosphorus'])
        K = float(request.form['potassium'])
        ph = float(request.form['ph_level'])
        area = float(request.form['area'])
        month = request.form['sowing_month']
        district = request.form['district']
        season = request.form['season']

        season_col_map = {
            'Kharif': 'avg kharif',
            'Rabi': 'avg rabi',
            'Summer': 'summer avg',
            'Whole Year': 'whole year avg'
        }
        col = season_col_map.get(season, 'avg kharif')  

        temp_value = data.loc[data['DISTRICT'] == district, month].values[0]
        rainfall_value = data.loc[data['DISTRICT'] == district, col].values[0]
        season_value = season_index.get(season, -1)

        test_sample = np.array([[area, N, P, K, season_value, rainfall_value, temp_value, ph]])
        test_sample = scaler.transform(test_sample)

        prob = rf_model.predict_proba(test_sample)
        top_n_indices = np.argsort(prob[0])[-3:][::-1]  
        top_n_classes = [rf_model.classes_[i] for i in top_n_indices]

        predictions = [key for key, val in crop_dict.items() if val in top_n_classes]

        app.logger.info(f"Predictions: {predictions}")

        # Ensure 3 predictions (if less, fill with 'N/A')
        prediction1 = predictions[0] if len(predictions) > 0 else "N/A"
        prediction2 = predictions[1] if len(predictions) > 1 else "N/A"
        prediction3 = predictions[2] if len(predictions) > 2 else "N/A"

        return render_template('index(4).html', prediction1=prediction1, prediction2=prediction2, prediction3=prediction3)

    except Exception as e:
        app.logger.error(f"Error during prediction: {e}")
        return render_template('index(4).html', prediction1="Error", prediction2="Error", prediction3="Error")

if __name__ == '__main__':
    app.run(debug=True)
