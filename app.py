from flask import Flask,Response,jsonify, render_template ,logging,request,send_file
app = Flask(__name__)
import sys
import os
from flask_cors import CORS
import requests
from datetime import datetime, timedelta
import pandas as pd
from pandas import DataFrame, Series


import github3, json
import numpy as np

import matplotlib
matplotlib.use('Agg')  # Use the Agg backend for non-interactive environments
import matplotlib.pyplot as plt

# Initilize flask app
app = Flask(__name__)
# Handles CORS (cross-origin resource sharing)
CORS(app)

# Add response headers to accept all types of  requests
def build_preflight_response():
    response = make_response()
    response.headers.add("Access-Control-Allow-Origin", "*")
    response.headers.add("Access-Control-Allow-Headers", "Content-Type")
    response.headers.add("Access-Control-Allow-Methods",
                         "PUT, GET, POST, DELETE, OPTIONS")
    return response

# Modify response headers when returning to the origin
def build_actual_response(response):
    response.headers.set("Access-Control-Allow-Origin", "*")
    response.headers.set("Access-Control-Allow-Methods",
                         "PUT, GET, POST, DELETE, OPTIONS")
    return response

github_token = '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'
repos = ["openai/openai-cookbook", "openai/openai-python", "elastic/elasticsearch",
         "milvus-io/pymilvus", "SebastianM/angular-google-maps"]

@app.route('/')
def home():
    return {"message": 'Hello world!'}

#returns the Forks and Stars from the list of repos given through Github API
#the LSTM prediction for Requirement 11.1:
@app.route('/lstm/issuescreated/week/reponum=<int:reponum>')
def LSTMIssuesCreatedWeek(reponum):
    df = pd.read_csv(f'datamodel/github_issues_data_repo{reponum}.csv')
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    df.sort_index(inplace=True)

    # Filter data for the last 4 months
    history_data = df[df.index >= (pd.to_datetime('today') - pd.DateOffset(months=24))]

    # Find the day of the week with the maximum number of issues in the last 2 months
    issues_last_2_months = df[df.index >= (pd.to_datetime('today') - pd.DateOffset(months=1))]
    max_day_of_week = issues_last_2_months.groupby(issues_last_2_months.index.day_name())['issues'].sum().idxmax()

    # LSTM model preparation
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(np.array(history_data['issues']).reshape(-1, 1))

    def create_dataset(dataset, time_steps=1):
        dataX, dataY = [], []
        for i in range(len(dataset)-time_steps):
            a = dataset[i:(i+time_steps), 0]
            dataX.append(a)
            dataY.append(dataset[i + time_steps, 0])

    time_steps = 7
    X, y = create_dataset(scaled_data, time_steps)

    # Reshape the data to be 3-dimensional for LSTM input
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # Build and train LSTM model
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], 1)))
    model.add(LSTM(units=50))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model
    history = model.fit(X_train, y_train, epochs=100, batch_size=8, validation_data=(X_test, y_test), shuffle=False)

    # Plot model loss graph for train loss and test loss
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Test Loss')
    plt.title('Model Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Forecast for the last 2 months
    test_data = scaled_data[-time_steps:]
    test_data = test_data.reshape(1, time_steps, 1)
    forecast = []

    for i in range(len(issues_last_2_months)):
        predicted_value = model.predict(test_data)
        forecast.append(predicted_value[0, 0])
        test_data = np.append(test_data[:, 1:, :], [[[predicted_value[0, 0]]]], axis=1)

    forecast = scaler.inverse_transform(np.array(forecast).reshape(-1, 1))

    # Visualize LSTM generated data showing three lines - history (4), true (last 2 months), and prediction lines
    plt.figure(figsize=(12, 6))
    plt.plot(history_data.index, history_data['issues'], label='History', color='green')
    plt.plot(issues_last_2_months.index, issues_last_2_months['issues'], label='True (last 2 months)', color='yellow')
    plt.plot(issues_last_2_months.index, forecast[-len(issues_last_2_months):], label='Prediction', linestyle='dashed', color='red')
    plt.title('LSTM Generated Data - History, True, and Prediction')
    plt.xlabel('Date')
    plt.ylabel('Number of Issues')
    plt.legend()

    return plt, 200

@app.route('/fetch/lstm/8.1/reponum=<int:reponum>')
def fetch_lstm_createdweek(reponum):
    # Specify the path to your image file
    image_path = f'LSTM/8.1-{reponum}.png'

    # Return the image as a response
    return send_file(image_path, mimetype='image/png')

@app.route('/fetch/lstm/8.2/reponum=<int:reponum>')
def fetch_lstm_two(reponum):
    # Specify the path to your image file
    image_path = f'LSTM/8.2-{reponum}.png'

    # Return the image as a response
    return send_file(image_path, mimetype='image/png')

@app.route('/fetch/lstm/8.3/reponum=<int:reponum>')
def fetch_lstm_three(reponum):
    # Specify the path to your image file
    image_path = f'LSTM/8.3-{reponum}.png'

    # Return the image as a response
    return send_file(image_path, mimetype='image/png')

@app.route('/fetch/lstm/8.4/reponum=<int:reponum>')
def fetch_lstm_4(reponum):
    # Specify the path to your image file
    image_path = f'LSTM/8.4-{reponum}.png'

    # Return the image as a response
    return send_file(image_path, mimetype='image/png')
@app.route('/fetch/lstm/8.5/reponum=<int:reponum>')
def fetch_lstm_5(reponum):
    # Specify the path to your image file
    image_path = f'LSTM/8.5-{reponum}.png'

    # Return the image as a response
    return send_file(image_path, mimetype='image/png')
@app.route('/fetch/lstm/8.6/reponum=<int:reponum>')
def fetch_lstm_6(reponum):
    # Specify the path to your image file
    image_path = f'LSTM/8.6-{reponum}.png'

    # Return the image as a response
    return send_file(image_path, mimetype='image/png')
@app.route('/fetch/lstm/8.7/reponum=<int:reponum>')
def fetch_lstm_7(reponum):
    # Specify the path to your image file
    image_path = f'LSTM/8.7-{reponum}.png'

    # Return the image as a response
    return send_file(image_path, mimetype='image/png')
@app.route('/fetch/lstm/8.8/reponum=<int:reponum>')
def fetch_lstm_8(reponum):
    # Specify the path to your image file
    image_path = f'LSTM/8.8-{reponum}.png'

    # Return the image as a response
    return send_file(image_path, mimetype='image/png')
@app.route('/fetch/lstm/8.9/reponum=<int:reponum>')
def fetch_lstm_9(reponum):
    # Specify the path to your image file
    image_path = f'LSTM/8.9-{reponum}.png'

    # Return the image as a response
    return send_file(image_path, mimetype='image/png')
@app.route('/fetch/lstm/8.10/reponum=<int:reponum>')
def fetch_lstm_10(reponum):
    # Specify the path to your image file
    image_path = f'LSTM/8.10-{reponum}.png'

    # Return the image as a response
    return send_file(image_path, mimetype='image/png')


#----------- Stats Model -------------------
@app.route('/fetch/sm/8.1/reponum=<int:reponum>')
def fetch_sm_one(reponum):
    # Specify the path to your image file
    image_path = f'statsmodel/8.1-{reponum}.png'

    # Return the image as a response
    return send_file(image_path, mimetype='image/png')

@app.route('/fetch/sm/8.2/reponum=<int:reponum>')
def fetch_sm_2(reponum):
    # Specify the path to your image file
    image_path = f'statsmodel/8.2-{reponum}.png'

    # Return the image as a response
    return send_file(image_path, mimetype='image/png')
@app.route('/fetch/sm/8.3/reponum=<int:reponum>')
def fetch_sm_3(reponum):
    # Specify the path to your image file
    image_path = f'statsmodel/8.3-{reponum}.png'

    # Return the image as a response
    return send_file(image_path, mimetype='image/png')
@app.route('/fetch/sm/8.4/reponum=<int:reponum>')
def fetch_sm_4(reponum):
    # Specify the path to your image file
    image_path = f'statsmodel/8.4-{reponum}.png'

    # Return the image as a response
    return send_file(image_path, mimetype='image/png')
@app.route('/fetch/sm/8.5/reponum=<int:reponum>')
def fetch_sm_5(reponum):
    # Specify the path to your image file
    image_path = f'statsmodel/8.5-{reponum}.png'

    # Return the image as a response
    return send_file(image_path, mimetype='image/png')
@app.route('/fetch/sm/8.6/reponum=<int:reponum>')
def fetch_sm_6(reponum):
    # Specify the path to your image file
    image_path = f'statsmodel/8.6-{reponum}.png'

    # Return the image as a response
    return send_file(image_path, mimetype='image/png')
@app.route('/fetch/sm/8.7/reponum=<int:reponum>')
def fetch_sm_7(reponum):
    # Specify the path to your image file
    image_path = f'statsmodel/8.7-{reponum}.png'

    # Return the image as a response
    return send_file(image_path, mimetype='image/png')
@app.route('/fetch/sm/8.8/reponum=<int:reponum>')
def fetch_sm_8(reponum):
    # Specify the path to your image file
    image_path = f'statsmodel/8.8-{reponum}.png'

    # Return the image as a response
    return send_file(image_path, mimetype='image/png')
@app.route('/fetch/sm/8.9/reponum=<int:reponum>')
def fetch_sm_9(reponum):
    # Specify the path to your image file
    image_path = f'statsmodel/8.9-{reponum}.png'

    # Return the image as a response
    return send_file(image_path, mimetype='image/png')
@app.route('/fetch/sm/8.10/reponum=<int:reponum>')
def fetch_sm_10(reponum):
    # Specify the path to your image file
    image_path = f'statsmodel/8.10-{reponum}.png'

    # Return the image as a response
    return send_file(image_path, mimetype='image/png')

#----------- Prophet -------------------
@app.route('/fetch/ph/8.1/reponum=<int:reponum>')
def fetch_ph_one(reponum):
    # Specify the path to your image file
    image_path = f'prophet/8.1-{reponum}.png'

    # Return the image as a response
    return send_file(image_path, mimetype='image/png')
@app.route('/fetch/ph/8.2/reponum=<int:reponum>')
def fetch_ph_2(reponum):
    # Specify the path to your image file
    image_path = f'prophet/8.2-{reponum}.png'

    # Return the image as a response
    return send_file(image_path, mimetype='image/png')
@app.route('/fetch/ph/8.3/reponum=<int:reponum>')
def fetch_ph_3(reponum):
    # Specify the path to your image file
    image_path = f'prophet/8.3-{reponum}.png'

    # Return the image as a response
    return send_file(image_path, mimetype='image/png')
@app.route('/fetch/ph/8.4/reponum=<int:reponum>')
def fetch_ph_4(reponum):
    # Specify the path to your image file
    image_path = f'prophet/8.4-{reponum}.png'

    # Return the image as a response
    return send_file(image_path, mimetype='image/png')
@app.route('/fetch/ph/8.5/reponum=<int:reponum>')
def fetch_ph_5(reponum):
    # Specify the path to your image file
    image_path = f'prophet/8.5-{reponum}.png'

    # Return the image as a response
    return send_file(image_path, mimetype='image/png')
@app.route('/fetch/ph/8.6/reponum=<int:reponum>')
def fetch_ph_6(reponum):
    # Specify the path to your image file
    image_path = f'prophet/8.6-{reponum}.png'

    # Return the image as a response
    return send_file(image_path, mimetype='image/png')
@app.route('/fetch/ph/8.7/reponum=<int:reponum>')
def fetch_ph_7(reponum):
    # Specify the path to your image file
    image_path = f'prophet/8.7-{reponum}.png'

    # Return the image as a response
    return send_file(image_path, mimetype='image/png')
@app.route('/fetch/ph/8.8/reponum=<int:reponum>')
def fetch_ph_8(reponum):
    # Specify the path to your image file
    image_path = f'prophet/8.8-{reponum}.png'

    # Return the image as a response
    return send_file(image_path, mimetype='image/png')
@app.route('/fetch/ph/8.9/reponum=<int:reponum>')
def fetch_ph_9(reponum):
    # Specify the path to your image file
    image_path = f'prophet/8.9-{reponum}.png'

    # Return the image as a response
    return send_file(image_path, mimetype='image/png')
@app.route('/fetch/ph/8.10/reponum=<int:reponum>')
def fetch_ph_10(reponum):
    # Specify the path to your image file
    image_path = f'prophet/8.10-{reponum}.png'

    # Return the image as a response
    return send_file(image_path, mimetype='image/png')

#run server
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80)
