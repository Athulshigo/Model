from django.shortcuts import render
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Bidirectional, LSTM, Dense, Dropout
from datetime import datetime
from tensorflow.keras.models import load_model
# Create your views here.


def home(request):
    return render(request, "index.html")

def predict(request):
    if request.method == 'POST':
        print(request.POST['date'])
        
        #Prediction
        data = pd.read_csv("https://stockprediction.ams3.cdn.digitaloceanspaces.com/google.csv")
        
        data['Date'] = pd.to_datetime(data['Date'])
        data.set_index('Date', inplace=True)

        # Extract the 'Close' prices
        closing_prices = data['Close'].values.reshape(-1, 1)

        # Features to be scaled
        features_to_scale = ['Open', 'High', 'Low', 'Volume', 'Adj Close']

        # Initialize MinMaxScaler for features
        feature_scaler = MinMaxScaler(feature_range=(0, 1))

        # Scale the features
        scaled_features = feature_scaler.fit_transform(data[features_to_scale])

        # Normalize the 'Close' prices
        scaler = MinMaxScaler(feature_range=(0, 1))
        closing_prices_scaled = scaler.fit_transform(closing_prices)

        # Create sequences for LSTM
        sequence_length = 30

        loaded_model = load_model("static/stock_price_model.h5")

        user_date_str = request.POST['date']
        user_date = datetime.strptime(user_date_str, "%Y-%m-%d")

        # Check if the user-entered date is not in the dataset
        if user_date > data.index[-1]:


            # Extend the DataFrame index beyond the last observed date
            extended_index = pd.date_range(start=data.index.min(), end=user_date, freq='D')

            # Reindex the DataFrame with the extended index
            extended_data = data.reindex(extended_index)

            # Linear interpolation for all features
            interpolated_data = extended_data.interpolate(method='spline', order=3)

            # Create input sequence directly from interpolated data
            input_sequence = interpolated_data.loc[:user_date, features_to_scale].values[-sequence_length:]

            # Continue with the rest of the code (unchanged)
            input_sequence_scaled = feature_scaler.transform(input_sequence)
            input_sequence_reshaped = np.reshape(input_sequence_scaled, (1, sequence_length, input_sequence.shape[1]))

            # Predict using the model
            predicted_price_scaled = loaded_model.predict(input_sequence_reshaped,verbose=False)
            predicted_price = scaler.inverse_transform(np.array([[predicted_price_scaled[0, -1]]]))[0, 0]

            print(f'Predicted Close Price for {user_date_str}: ${predicted_price:.2f}')

        else:
            # Preprocess user input for regular prediction
            input_sequence = data.loc[:user_date, features_to_scale].values[-sequence_length:]
            input_sequence_scaled = feature_scaler.transform(input_sequence)
            input_sequence_reshaped = np.reshape(input_sequence_scaled, (1, sequence_length, input_sequence.shape[1]))

            # Predict using the model
            predicted_price_scaled = loaded_model.predict(input_sequence_reshaped,verbose=False)
            predicted_price = scaler.inverse_transform(np.array([[predicted_price_scaled[0, 0]]]))[0, 0]

            print(f'Predicted Close Price for {user_date_str}: ${predicted_price:.2f}')



    return render(request, "Predicted.html", {'predicted':round(predicted_price, 2), 'date': request.POST['date']})