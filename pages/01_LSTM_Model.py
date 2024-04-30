import pandas as pd
import numpy as np
import tensorflow as tf
import random as rn
import yfinance as yf
import streamlit as st
import datetime as dt
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
np.random.seed(1)
tf.random.set_seed(1)
rn.seed(1)
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from millify import millify
import pandas_datareader as web
from cryptocmd import CmcScraper

# Function to format large numbers
def format_number(number):
    return '{:,.2f}'.format(number)

# Streamlit app title
st.title('Next-Day Forecasting with Long-Short Term Memory (LSTM)')

# Load cryptocurrency symbols from CSV
csv = pd.read_csv('convertcsv.csv')
symbol = csv['symbol'].tolist()

# Sidebar for selecting cryptocurrency and date range
st.sidebar.subheader('Select Cryptocurrency and Date Range')
ticker_input = st.sidebar.selectbox('Choose a Cryptocurrency', symbol, index=symbol.index('ETH'))
start_date = st.sidebar.date_input('Start Date', dt.datetime.today() - dt.timedelta(5*365))
end_date = st.sidebar.date_input('End Date', dt.datetime.today())

# Initialize scraper with selected cryptocurrency and date range
scraper = CmcScraper(ticker_input, start_date.strftime('%d-%m-%Y'), end_date.strftime('%d-%m-%Y'))
df = scraper.get_dataframe()

# Display loading message
st.write('Fitting the model. Please wait...')

# Sort dataframe by date
eth_df = df.sort_values(['Date'], ascending=True, axis=0)

# Create dataframe for LSTM model
eth_lstm = pd.DataFrame(index=range(0, len(eth_df)), columns=['Date', 'Close'])
for i in range(0, len(eth_df)):
    eth_lstm['Date'][i] = eth_df['Date'][i]
    eth_lstm['Close'][i] = eth_df['Close'][i]

eth_lstm.index = eth_lstm.Date
eth_lstm.drop('Date', axis=1, inplace=True)
eth_lstm = eth_lstm.sort_index(ascending=True)

# Split dataset into train and test sets
dataset = eth_lstm.values
train = dataset[0:990, :]
valid = dataset[990:, :]

# Scale the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)

# Prepare data for LSTM model
x_train, y_train = [], []
for i in range(60, len(train)):
    x_train.append(scaled_data[i-60:i, 0])
    y_train.append(scaled_data[i, 0])

x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

# Create LSTM model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(LSTM(units=50))
model.add(Dense(1))

model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(x_train, y_train, epochs=1, batch_size=1, verbose=2)

# Prepare test data for prediction
inputs = eth_lstm[len(eth_lstm) - len(valid) - 60:].values
inputs = inputs.reshape(-1, 1)
inputs = scaler.transform(inputs)

X_test = []
for i in range(60, inputs.shape[0]):
    X_test.append(inputs[i-60:i, 0])

X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
closing_price = model.predict(X_test)
closing_price = scaler.inverse_transform(closing_price)

# Calculate metrics
rms = np.sqrt(mean_squared_error(closing_price, valid))
acc = r2_score(closing_price, valid) * 100

# Plot actual vs. predicted prices
train = eth_df[:990]
valid = eth_df[990:]
valid['Predictions'] = closing_price

st.subheader('Actual vs Predicted Prices')
fig_preds = go.Figure()

fig_preds.add_trace(
    go.Scatter(
        x=train['Date'],
        y=train['Close'],
        name='Training Data Closing Price',
        line=dict(color='royalblue', width=2)
    )
)

fig_preds.add_trace(
    go.Scatter(
        x=valid['Date'],
        y=valid['Close'],
        name='Validation Data Closing Price',
        line=dict(color='firebrick', width=2)
    )
)

fig_preds.add_trace(
    go.Scatter(
        x=valid['Date'],
        y=valid['Predictions'],
        name='Predicted Closing Price',
        line=dict(color='forestgreen', width=2)
    )
)

fig_preds.update_layout(
    legend=dict(
        orientation='h',
        yanchor='bottom',
        y=1,
        xanchor='left',
        x=0),
    height=600,
    title_text='Predictions on Validation Data',
    template='plotly_dark',
    xaxis=dict(title='Date'),
    yaxis=dict(title='Price')
)

st.plotly_chart(fig_preds, use_container_width=True)

# Display metrics
mae = mean_absolute_error(closing_price, valid['Close'])
rmse = np.sqrt(mean_squared_error(closing_price, valid['Close']))
accuracy = r2_score(closing_price, valid['Close']) * 100

st.subheader('Metrics')
st.write(f'Absolute Error between Predicted and Actual Value: {format_number(mae)}')
st.write(f'Root Mean Squared Error between Predicted and Actual Value: {format_number(rmse)}')
st.write(f'Accuracy of the Model: {format_number(accuracy)} %')

# Forecasting for the next day
real_data = [inputs[len(inputs) - 60:len(inputs + 1), 0]]
real_data = np.array(real_data)
real_data = np.reshape(real_data, (real_data.shape[0], real_data.shape[1], 1))

prediction = model.predict(real_data)
prediction = scaler.inverse_transform(prediction)

st.subheader('Next-Day Forecasting')
st.write(f'Closing Price Prediction of the Next Trading Day for {ticker_input}: $ {format_number(float(prediction))}')
