#!/usr/bin/env python
# coding: utf-8

# In[4]:


#!pip install pandas-ta
#!pip install scikeras
#!pip install nbformat


# In[2]:


import pandas as pd
import pandas_ta as ta
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.model_selection import GridSearchCV
from scikeras.wrappers import KerasRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import mean_squared_error,mean_absolute_percentage_error,mean_absolute_error
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score,f1_score
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# In[3]:


file_list = ["2018_Global_Markets_Data.csv",
             "2019_Global_Markets_Data.csv",
             "2020_Global_Markets_Data.csv",
             "2021_Global_Markets_Data.csv",
             "2022_Global_Markets_Data.csv",
             "2023_Global_Markets_Data.csv"]
data_frames = []

for file in file_list :
    df = pd.read_csv(file)
    data_frames.append(df)

merged_df = pd.concat(data_frames, ignore_index=True)

merged_df.drop("Adj Close", axis=1, inplace=True)

merged_df["Date"] = pd.to_datetime(merged_df["Date"])
merged_df.sort_values(by="Date", inplace=True)

merged_df.to_csv("Total_Global_Markets_Data.csv", index=False)


# In[5]:


df = merged_df[merged_df['Ticker'] == '^NSEI']
df.ta.rsi(close=df["Close"], length=14, append=True)
df.ta.macd(close=df["Close"], fast=12, slow=26, signal=9, append=True)
df


# In[6]:


years = df['Date'].dt.year.unique()
for year in years:
    year_data = df[df['Date'].dt.year == year]

    fig = go.Figure(data=[go.Candlestick(x=year_data['Date'], open=year_data['Open'],
                    high=year_data['High'], low=year_data['Low'], close=year_data['Close'])])
    fig.update_layout(
        title=f'Candlestick Chart - Year {year}', xaxis_title='Date', yaxis_title='Price')
    fig.show()

    fig = go.Figure(data=[go.Bar(x=year_data['Date'], y=year_data['Volume'],
                    text=year_data['Volume'], textposition='outside', marker_color='gold')])
    fig.update_layout(
        title=f'Volume - Year {year}', xaxis_title='Date', yaxis_title='Volume')
    fig.show()

    fig = go.Figure(data=[go.Scatter(x=year_data["Date"], y=year_data["RSI_14"],
                    mode="lines", name="RSI", line=dict(color="blue"))])
    fig.add_hline(y=70, line_dash="dash", line=dict(color='red'),
                  annotation_text='Overbought (70)', row=1, col=1)
    fig.add_hline(y=30, line_dash="dash", line=dict(color='green'),
                  annotation_text='Oversold (30)', row=1, col=1)
    fig.update_layout(
        title=f'RSI - Year {year}', xaxis_title='Date', yaxis_title='RSI')
    fig.show()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=year_data["Date"], y=year_data["MACD_12_26_9"],
                  mode="lines", name="MACD", line=dict(color="green")))
    fig.add_trace(go.Scatter(x=year_data["Date"], y=year_data["MACDs_12_26_9"],
                  mode="lines", name="Signal", line=dict(color="red")))
    fig.add_trace(go.Bar(x=year_data["Date"], y=year_data["MACDh_12_26_9"], name="Histogram", marker_color=np.where(
        year_data["MACDh_12_26_9"] < 0, "red", "green")))
    fig.update_layout(
        title=f'MACD - Year {year}', xaxis_title='Date', yaxis_title='RSI')
    fig.show()


# In[7]:


years = df['Date'].dt.year.unique()
for year in years:
    year_data = df[df['Date'].dt.year == year]

    fig = go.Figure(data=[go.Candlestick(x=year_data['Date'], open=year_data['Open'],
                    high=year_data['High'], low=year_data['Low'], close=year_data['Close'])])
    fig.update_layout(
        title=f'Candlestick Chart - Year {year}', xaxis_title='Date', yaxis_title='Price')
    fig.show()


# In[8]:


years = df['Date'].dt.year.unique()
for year in years:
    year_data = df[df['Date'].dt.year == year]

    fig = go.Figure(data=[go.Bar(x=year_data['Date'], y=year_data['Volume'],
                    text=year_data['Volume'], textposition='outside', marker_color='gold')])
    fig.update_layout(
        title=f'Volume - Year {year}', xaxis_title='Date', yaxis_title='Volume')
    fig.show()


# In[9]:


years = df['Date'].dt.year.unique()
for year in years:
    year_data = df[df['Date'].dt.year == year]
    fig = go.Figure(data=[go.Scatter(x=year_data["Date"], y=year_data["RSI_14"],
                    mode="lines", name="RSI", line=dict(color="blue"))])
    fig.add_hline(y=70, line_dash="dash", line=dict(color='red'),
                  annotation_text='Overbought (70)', row=1, col=1)
    fig.add_hline(y=30, line_dash="dash", line=dict(color='green'),
                  annotation_text='Oversold (30)', row=1, col=1)
    fig.update_layout(
        title=f'RSI - Year {year}', xaxis_title='Date', yaxis_title='RSI')
    fig.show()


# In[10]:


years = df['Date'].dt.year.unique()
for year in years:
    year_data = df[df['Date'].dt.year == year]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=year_data["Date"], y=year_data["MACD_12_26_9"],
                  mode="lines", name="MACD", line=dict(color="green")))
    fig.add_trace(go.Scatter(x=year_data["Date"], y=year_data["MACDs_12_26_9"],
                  mode="lines", name="Signal", line=dict(color="red")))
    fig.add_trace(go.Bar(x=year_data["Date"], y=year_data["MACDh_12_26_9"], name="Histogram", marker_color=np.where(
        year_data["MACDh_12_26_9"] < 0, "red", "green")))
    fig.update_layout(
        title=f'MACD - Year {year}', xaxis_title='Date', yaxis_title='RSI')
    fig.show()


# In[11]:


nsei_data = merged_df[merged_df['Ticker']
                      == '^NSEI'][['Date', 'Ticker', 'Close']]
nsei_data.to_csv('NSEI_data.csv', index=False)


# In[12]:


# Normalize the data
scaler = MinMaxScaler()


# In[13]:


# Define the sequence length
seq_length = 5

# Define a function to create sequences and labels


def create_sequences(data, seq_length):
    sequences = []
    labels = []
    for i in range(len(data) - seq_length):
        sequences.append(data[i:i + seq_length])
        labels.append(data[i + seq_length])
    return np.array(sequences), np.array(labels)


# In[14]:


# Define a function to build the LSTM model
def build_lstm_model(units=50, dropout=0.1):
    model = Sequential()
    model.add(LSTM(units=units, return_sequences=True,
              input_shape=(seq_length, 1)))
    model.add(Dropout(dropout))
    model.add(LSTM(units=units))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model


# In[15]:


# Normalize the data
normalized_data_nsei = scaler.fit_transform(
    nsei_data['Close'].values.reshape(-1, 1))

# Split data into training and testing sets
train_size_nsei = int(len(normalized_data_nsei) * 0.8)
train_data_nsei, test_data_nsei = normalized_data_nsei[:
                                                       train_size_nsei], normalized_data_nsei[train_size_nsei:]

# Create sequences and labels
train_sequences_nsei, train_labels_nsei = create_sequences(
    train_data_nsei, seq_length)
test_sequences_nsei, test_labels_nsei = create_sequences(
    test_data_nsei, seq_length)

# Create a KerasRegressor wrapper for scikit-learn compatibility
model_nsei = KerasRegressor(model=build_lstm_model, units=5, dropout=0.1)

# Define the parameter grid
param_grid_nsei = {'batch_size': [16, 32], 'epochs': [
    5, 10], 'units': [5, 7], 'dropout': [0.1, 0.2]}

# Create the GridSearchCV object
grid_search_nsei = GridSearchCV(estimator=model_nsei, param_grid=param_grid_nsei, scoring=[
                                'r2', 'explained_variance', 'neg_mean_squared_error'], cv=3, refit='r2', verbose=1)

# Fit the search to training data
grid_search_nsei.fit(train_sequences_nsei, train_labels_nsei)

# Get the best parameters
best_params_nsei = grid_search_nsei.best_params_

# Train the final model with best parameters
best_model_nsei = build_lstm_model(
    units=best_params_nsei['units'], dropout=best_params_nsei['dropout'])
best_model_nsei.fit(train_sequences_nsei, train_labels_nsei,
                    epochs=best_params_nsei['epochs'], batch_size=best_params_nsei['batch_size'], verbose=0)

# Make predictions and plot for each feature
fig_nsei = make_subplots(rows=1, cols=1)

test_predictions_nsei = best_model_nsei.predict(test_sequences_nsei)
test_predictions_nsei = scaler.inverse_transform(test_predictions_nsei)
test_actual_nsei = scaler.inverse_transform(test_labels_nsei)

mse_nsei = mean_squared_error(test_actual_nsei, test_predictions_nsei)
mae_nsei = mean_absolute_error(test_actual_nsei, test_predictions_nsei)
rmse_nsei = np.sqrt(mse_nsei)
r2_nsei = r2_score(test_actual_nsei, test_predictions_nsei)
print(f"MSE: {mse_nsei:.2f}, MAE: {mae_nsei:.2f}, RMSE: {rmse_nsei:.2f}, R-squared: {r2_nsei:.2f}")

fig_nsei.add_trace(go.Scatter(x=nsei_data['Date'][train_size_nsei+seq_length:],
                   y=test_actual_nsei.ravel(), mode='lines', name=f'Actual'), row=1, col=1)
fig_nsei.add_trace(go.Scatter(x=nsei_data['Date'][train_size_nsei+seq_length:],
                   y=test_predictions_nsei.ravel(), mode='lines', name=f'Predicted'), row=1, col=1)
fig_nsei.update_layout(
    title='LSTM Time Series Forecasting for ^NSEI', height=800)
fig_nsei.show()


# In[17]:


# Create a date range from 31-07-2023 to 31-12-2024
start_date = '2023-07-31'
end_date = '2023-12-31'
date_range = pd.date_range(start=start_date, end=end_date, freq='D')

# Filter out weekends (Saturday and Sunday)
business_days = date_range[date_range.weekday < 5]

# Create a DataFrame with the filtered dates
nsei_prediction_df = pd.DataFrame({'Date': business_days})

nsei_prediction_df['Ticker'] = '^NSEI'


# In[18]:


# Normalize the data
scaler = MinMaxScaler()


# In[19]:


nsei_final_data = pd.read_csv('NSEI_data.csv')

# Normalize the data
nsei_normalized_data = scaler.fit_transform(
    nsei_final_data['Close'].values.reshape(-1, 1))

# Create sequences and labels
nsei_train_sequences, nsei_train_labels = create_sequences(
    nsei_normalized_data, seq_length)

nsei_test_sequence = nsei_train_labels[-seq_length:]

model = best_model_nsei

nsei_prediction_data = np.empty((0,))
for i in range(110):
    # Reshape the test sequence to match the input shape expected by the model
    nsei_test_sequence = nsei_test_sequence.reshape(1, -1, 1)
    nsei_prediction = model.predict(nsei_test_sequence)
    nsei_prediction_data = np.append(nsei_prediction_data, nsei_prediction)
    nsei_test_sequence = np.append(nsei_test_sequence, nsei_prediction)
    nsei_test_sequence = nsei_test_sequence[1:]

# Reshape and inverse transform the predictions
nsei_prediction_data = nsei_prediction_data.reshape(-1, 1)
nsei_prediction_data = scaler.inverse_transform(nsei_prediction_data)
nsei_prediction_data = nsei_prediction_data.ravel()

nsei_prediction_df['Close'] = nsei_prediction_data

nsei_prediction_df


# In[20]:


nsei_prediction_df.to_csv('NSEI_prediction.csv', index=False)


# In[21]:


nsei_total_df = pd.concat([pd.read_csv('NSEI_data.csv'), pd.read_csv(
    'NSEI_prediction.csv')], ignore_index=True)
nsei_total_df["Date"] = pd.to_datetime(nsei_total_df["Date"])
nsei_total_df.sort_values(by="Date", inplace=True)

nsei_total_df = nsei_total_df[nsei_total_df['Date'].dt.year == 2023]

split_date = pd.to_datetime("28-07-2023")
nsei_before_split = nsei_total_df[nsei_total_df['Date'] <= split_date]
nsei_after_split = nsei_total_df[nsei_total_df['Date'] > split_date]


# In[22]:


fig_final = go.Figure()
fig_final.add_trace(go.Scatter(
    x=nsei_before_split["Date"], y=nsei_before_split["Close"], mode="lines", name="Provided", line=dict(color="blue")))
fig_final.add_trace(go.Scatter(
    x=nsei_after_split["Date"], y=nsei_after_split["Close"], mode="lines", name="Predicted", line=dict(color="red")))
fig_final.update_layout(title='LSTM Closing Price Prediction ^NSEI',
                  xaxis_title='Date', yaxis_title='Price')
fig_final.show()


# In[26]:


get_ipython().system('pip install pmdarima')
from statsmodels.tsa.arima_model import ARIMA
from pmdarima.arima import auto_arima


# In[28]:


# Create a DataFrame with the non-stationary data
data = pd.DataFrame({'close': nsei_data['Close']})

# Function to make time series stationary by differencing
def make_stationary(data, diff_order=1):
    # Perform differencing
    stationary_data = np.diff(data, n=diff_order)

    return pd.Series(stationary_data)

# Make the column stationary by differencing
stationary_temp = make_stationary(nsei_data['Close'])

# Plot original non-stationary data
plt.figure(figsize=(10, 6))
plt.plot(nsei_data['Close'], label='Non-Stationary Data')
plt.title('Non-Stationary  Data')
plt.legend()
plt.show()

# Plot differenced stationary data
plt.figure(figsize=(10, 6))
plt.plot(stationary_temp, label='Stationary Data')
plt.title('Stationary Time Series Data After Differencing')
plt.legend()
plt.show()

# Apply ADF test on the differenced 'Close' column
ad_test_result = adfuller(stationary_temp)
print("ADF Statistic:", ad_test_result[0])
print("P-Value:", ad_test_result[1])
print("Critical Values:", ad_test_result[4])


# In[29]:


stepwise_fit = auto_arima(stationary_temp,trace=True,suppress_warnings=True)

stepwise_fit.summary()


# In[30]:


train=stationary_temp.iloc[:-40]
test=stationary_temp.iloc[-40:]
print(train.shape,test.shape)


# In[31]:


from statsmodels.tsa.arima.model import ARIMA
model = ARIMA(train, order=(0,0,0))
model_fit = model.fit()
# Display the model summary
print(model_fit.summary())


# In[32]:


# Define the start and end indices for prediction
start = len(train)
end = len(train) + len(test) - 1

# Predict using the fitted model
pred = model_fit.predict(start=start, end=end, typ='levels')
pred.index=df.index[start:end+1]

# Display the predictions
print(pred)


# In[33]:


from sklearn.metrics import mean_squared_error,mean_absolute_percentage_error,mean_absolute_error


# In[34]:


#evaluating the model
mae = mean_absolute_error(pred,test)
print('MAE:', mae)
mse = mean_squared_error(pred,test)
print('MSE:', mse)
rmse = np.sqrt(mean_squared_error(pred,test))
print('RMSE:', rmse)
mape = mean_absolute_percentage_error(pred,test)
print('MAPE:', mape)


# In[ ]:




