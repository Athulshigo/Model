from django.shortcuts import render
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from datetime import datetime
import joblib
# Create your views here.


def home(request):
    return render(request, "index.html")


def predict(request):
    if request.method == 'POST':
        print(request.POST['date'])
        
        #Prediction
        df = pd.read_csv("https://next-tail-space.blr1.digitaloceanspaces.com/stockprice/google.csv", parse_dates=['Date'])
        df = df.sort_values(by='Date')  # Sort the DataFrame by date
        features = ['High', 'Low', 'Price', 'Volume', 'Adj Close', 'Close']
        X = df[features]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        loaded_model = joblib.load("static/stock_price_model.joblib")
        user_input_date = request.POST['date']
        most_recent_data = df[df['Date'] <= user_input_date].iloc[-1]
        user_input_features = most_recent_data[features].values.reshape(1, -1)
        user_input_features_scaled = scaler.transform(user_input_features)
        predicted_close = loaded_model.predict(user_input_features_scaled)
        print(f"Predicted Close Price for {user_input_date}: {predicted_close[0]}")


    return render(request, "Predicted.html", {'predicted':round(predicted_close[0], 2), 'date': request.POST['date']})