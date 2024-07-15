# app.py
from flask import Flask, request, jsonify
import json
import joblib
import numpy as np
from flask_cors import CORS

app = Flask(__name__)

CORS(app)

# Load the model and encoders
model = joblib.load('salary_model.pkl')
label_encoder_region = joblib.load('label_encoder_region.pkl')
label_encoder_jobRole = joblib.load('label_encoder_jobRole.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    region = data['region']
    jobRole = data['jobRole']

    # Encode the input using the same label encoders
    region_encoded = label_encoder_region.transform([region])[0]
    jobRole_encoded = label_encoder_jobRole.transform([jobRole])[0]

    input_data = np.array([[region_encoded, jobRole_encoded]])
    predicted_salary = model.predict(input_data)[0]
    app.logger.info(json.dumps({'predicted_salary': predicted_salary}))
    return json.dumps({'predicted_salary': predicted_salary})

if __name__ == '__main__':
    app.run(debug=True)
