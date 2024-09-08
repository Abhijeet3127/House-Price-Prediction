from flask import Flask, request, jsonify
import joblib
import numpy as np

# Load the trained model
model = joblib.load('house_price_model.pkl')

# Initialize the Flask app
app = Flask(__name__)

# Define the predict route
@app.route('/predict', methods=['POST'])
def predict():
    # Get data from the POST request
    data = request.get_json(force=True)
    
    # Extract features from the request data
    features = [
        data['Lot Area'],
        data['Overall Qual'],
        data['Overall Cond'],
        data['Year Built'],
        data['Gr Liv Area'],
        data['Total Bsmt SF'],
        data['Garage Area'],
        data['Full Bath'],
        data['Half Bath'],
        data['Bedroom AbvGr'],
        data['TotRms AbvGrd']
    ]

    # Convert features to a numpy array for prediction
    features_array = np.array([features])
    
    # Make the prediction using the loaded model
    prediction = model.predict(features_array)[0]
    
    # Return the prediction as a JSON response
    return jsonify({'predicted_price': round(prediction, 2)})

if __name__ == '__main__':
    app.run(debug=True)
