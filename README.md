# House Price Prediction API using Random Forest and Flask

This project demonstrates how to build a machine learning model using Random Forest to predict house prices based on various features from the Ames Housing dataset. The model is then deployed as a REST API using Flask.

## Features

- Trains a Random Forest Regressor to predict house prices.
- Provides an endpoint to get predictions using a JSON input.
- Example `curl` commands provided for testing.

## Requirements

To run this project, you need Python 3.x installed along with the following libraries:

- Flask
- NumPy
- Pandas
- Scikit-Learn
- Joblib

Install the dependencies using the following command:

```bash
pip install -r requirements.txt

## Run Flask App

To run the Flask app, execute the following command:

```bash
python main.py

## cURL

- First cURL

curl --location 'http://127.0.0.1:5000/predict' \
--header 'Content-Type: application/json' \
--data '{
  "Lot Area": 12000,
  "Overall Qual": 8,
  "Overall Cond": 5,
  "Year Built": 2005,
  "Gr Liv Area": 2500,
  "Total Bsmt SF": 1000,
  "Garage Area": 600,
  "Full Bath": 2,
  "Half Bath": 1,
  "Bedroom AbvGr": 3,
  "TotRms AbvGrd": 7
}'

- Second cURL

curl --location 'http://127.0.0.1:5000/predict' \
--header 'Content-Type: application/json' \
--data '{
  "Lot Area": 6000,
  "Overall Qual": 5,
  "Overall Cond": 6,
  "Year Built": 1975,
  "Gr Liv Area": 1200,
  "Total Bsmt SF": 500,
  "Garage Area": 300,
  "Full Bath": 1,
  "Half Bath": 0,
  "Bedroom AbvGr": 2,
  "TotRms AbvGrd": 5
}'


