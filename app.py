import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader.data as data
from sklearn.model_selection import train_test_split
from keras.models import load_model
import streamlit as st

start_date = '2010-01-01'
end_date = '2019-12-31'

st.title('Stock Trend Prediction')

user_input = st.text_input('Enter Stock Ticker','AAPL')

df = data.get_data_tiingo(user_input, api_key='cd8d4b275af91ab35f9bdd2ea6023790a946718a',start = start_date,end = end_date)

#Describing Data 
st.subheader('Data from 2010 to 2019')
st.write(df.describe())

#Visualizations
# st.subheader('Closing Price vs Time Chart')
# plt.plot(df.close.values)
# st.pyplot(plt.figure(figsize=(12, 6)))

st.subheader('Closing Price vs Time Chart')
plt.plot(df.close.values)  # Convert df.close to a NumPy array using .values
plt.xlabel('Time')
plt.ylabel('Closing Price')
plt.title('Closing Price vs Time')
st.pyplot(plt.gcf())

st.subheader('Closing Price vs Time Chart with 100MA')
ma100 = df.close.rolling(100).mean()  # Add parenthesis to calculate the rolling mean
plt.plot(ma100.values,'r')
plt.xlabel('Time')
plt.ylabel('Closing Price')
plt.title('Closing Price vs Time')
st.pyplot(plt.gcf())

st.subheader('Closing Price vs Time Chart with 100MA & 200MA')
ma100 = df.close.rolling(100).mean()  # Add parenthesis to calculate the rolling mean
ma200 = df.close.rolling(200).mean()
plt.plot(ma100.values,'r')
plt.plot(ma200.values,'g')
plt.xlabel('Time')
plt.ylabel('Closing Price')
plt.title('Closing Price vs Time')
st.pyplot(plt.gcf())


# Splitting data into training and testing
data_training = pd.DataFrame(df['close'][0:int(len(df)*0.70)])
data_testing = pd.DataFrame(df['close'][int(len(df)*0.70):int(len(df))])

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))

data_training_array = scaler.fit_transform(data_training)



#Load the model
model = load_model('keras_model.h5')

#Testing Part

past_100_days = data_training.tail(100)
final_df = past_100_days.append(data_testing,ignore_index = True)
input_data = scaler.fit_transform(final_df)

x_test = []
y_test = []

for i in range(100,input_data.shape[0]):
    x_test.append(input_data[i-100:i])
    y_test.append(input_data[i,0])

x_test,y_test = np.array(x_test),np.array(y_test)

# Making Predictions
y_predicted = model.predict(x_test)
scaler = scaler.scale_

scale_factor = 1/scaler[0]
y_predicted = y_predicted * scale_factor
y_test = y_test * scale_factor


#Final Graph
st.subheader('Predictions vs Original')
fig2 = plt.figure(figsize = (12,6))
plt.plot(y_test,'b',label = "Original Price")
plt.plot(y_predicted,'r',label = "Predicted Price")
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)