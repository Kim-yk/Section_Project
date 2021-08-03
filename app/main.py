from flask import Flask, render_template, request, jsonify, url_for
from flask_sqlalchemy import SQLAlchemy

import pandas as pd
import numpy as np
import yfinance as yf
import datetime as dt
from pandas_datareader import data as pdr

import matplotlib.pyplot as plt
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import LSTM,Dense

app = Flask(__name__)

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///stocks.db'
db = SQLAlchemy(app)

class Stock(db.Model):
    __tablename__ = 'Stock_info'
    id = db.Column(db.Integer, primary_key=True)
    stock_name = db.Column(db.String(80), unique=False)
    date = db.Column(db.String(120), unique=False)
    open = db.Column(db.String(120), unique=False)
    high = db.Column(db.Float, unique=False)
    low = db.Column(db.Float, unique=False)
    close = db.Column(db.Float, unique=False)
    adj_close = db.Column(db.Float, unique=False)
    volume = db.Column(db.Float, unique=False)

db.create_all()

def populate_database(stock_symbol, stock_data, db):
    df = stock_data

    for i in df.index:
        # insert the values into the database
        db.session.add(Stock(stock_name=stock_symbol, date=str(i), open=str(df['Open'][i]), high=float(df['High'][i]), low=float(df['Low'][i]), close=float(df['Close'][i]), adj_close=float(df['Adj Close'][i]), volume=float(df['Volume'][i])))
        db.session.commit() 

def main(ticker):
    stock_symbol = ticker

    #clear existing data
    db.session.query(Stock).delete()
    db.session.commit()

    # get the stock data
    stock_data = get_stock(stock_symbol)
    # populate the database
    populate_database(stock_symbol, stock_data, db)

def get_stock(stock_symbol):
    yf.pdr_override()

    start_year = 2020
    start_month = 1
    start_day = 1

    start = dt.datetime(start_year, start_month, start_day)
    now = dt.datetime.now()

    df = pdr.get_data_yahoo(stock_symbol, start, now)

    return df
def prediction_model(ticker):
  yf.pdr_override()

  start_year = 2020
  start_month = 1
  start_day = 1

  start = dt.datetime(start_year, start_month, start_day)
  now = dt.datetime.now()

  df = pdr.get_data_yahoo(ticker, start, now)
  df = df.sort_values(by=['Date'],ascending=False)

  high_prices = df['High'].values
  low_prices = df['Low'].values
  mid_prices = (high_prices+low_prices)/2

  seq_len = 50 
  sequence_length = seq_len + 1

  result = []

  for index in range(len(mid_prices)-sequence_length):
        result.append(mid_prices[index:index+sequence_length])

    
  normalized_data = []
  window_mean = []
  window_std = []

  for window in result:
      normalized_window = [((p - np.mean(window))/ np.std(window))for p in window]
      normalized_data.append(normalized_window)
      window_mean.append(np.mean(window))
      window_std.append(np.std(window))
    
  result = np.array(normalized_data)

    
  row = int(round(result.shape[0]*0.9))
  train = result[:row,:]
  np.random.shuffle(train)

  x_train = train[:,:-1]
  x_train = np.reshape(x_train,(train.shape[0], x_train.shape[1],1))
  y_train = train[:,-1]

  x_test = result[row:,:-1]
  x_test = np.reshape(x_test,(x_test.shape[0], x_test.shape[1],1))
  y_test = result[row:,-1]

  x_train.shape,x_test.shape    

  model = Sequential()
  model.add(LSTM(50, return_sequences=True, input_shape=(50,1)))
  model.add(LSTM(64,return_sequences=False))
  model.add(Dense(1,activation='linear')) 
  model.compile(loss='mse',optimizer='rmsprop')
  model.summary()
    
 
  model.fit(x_train,y_train,validation_data=(x_test,y_test),batch_size=10,epochs=30)
    
  pred = model.predict(x_test)
    
  pred_result = []
  pred_y = []
  for i in range(len(pred)):
      n1 = (pred[i] * window_std[i]) + window_mean[i]
      n2 = (y_test[i] * window_std[i]) + window_mean[i]
      pred_result.append(n1)
      pred_y.append(n2)

  fig = plt.figure(facecolor='white', figsize=(20, 10))
  ax = fig.add_subplot(111)
  ax.plot(pred_result, label='Prediction')
  ax.legend()      
  return ax

@app.route('/')
def home():
    return render_template('home.html') 

@app.route('/stocks', methods=['GET','POST'])
def stocks():
    if request.method == 'POST':
        ticker = request.form.get('ticker-symbol')
        main(ticker)
        return render_template('stocks.html',values=Stock.query.all())
    else:
        return render_template('stocks.html',values=Stock.query.all())
   

@app.route('/prediction',methods=['GET','POST'])
def prediction(ticker):
    ax = prediction_model(ticker)
    return render_template('prediction.html',plot_url=ax)

  
  
#run the program by running this app
if __name__ == "__main__":
    app.run(debug=True)
