import joblib
import sklearn
import pandas as pd
import warnings

warnings.filterwarnings('ignore')
climate_data = pd.read_csv('merged.csv')

rf_model = joblib.load('rf_model.pkl')
scaler = joblib.load('scaler.pkl')

feature_columns = ['Area_in_hectares', 'N', 'P', 'K', 'Crop_Type_labels', 'rainfall',
 'temperature', 'pH']

district = 'BULDHANA'

test_sample = [[200.0, 10, 40, 40, 1, 444.34, 16.26, 6.0]]
test_sample_df = pd.DataFrame(test_sample, columns=feature_columns)
new_samples_scaled = scaler.transform(test_sample_df)
prediction = rf_model.predict(new_samples_scaled)

print('prediction: ', prediction)