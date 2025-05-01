# -*- coding: utf-8 -*-
"""
Created on Mon Apr  7 17:50:40 2025

@author: david
"""

import tensorflow as tf
import datetime as dt
import pandas as pd
import numpy as np
import yfinance as yf

stocks = ['NVDA']
days= 180

end_date = dt.datetime.now() 
start_date = end_date - dt.timedelta(days=days)

stockdata= yf.download(stocks, start_date, end_date)
stockdata = stockdata['Close']

n = len(stockdata)
train_df = stockdata[:int(n*0.7)]
val_df = stockdata[int(n*0.7):int(n*0.9)]
test_df = stockdata[int(n*0.9):]

# split_index = int(len(stockdata) * 0.8)
# set_train = stockdata.iloc[:split_index]
# set_test = stockdata.iloc[split_index:]

train_mean = train_df.mean()
train_std = train_df.std()

train_df = (train_df - train_mean) / train_std
val_df = (val_df - train_mean) / train_std
test_df = (test_df - train_mean) / train_std


class WindowGenerator():
  def __init__(self, input_width, label_width, shift,
               train_df=train_df, val_df=val_df, test_df=test_df,
               label_columns=None):
    # Store the raw data.
    self.train_df = train_df
    self.val_df = val_df
    self.test_df = test_df

    # Work out the label column indices.
    self.label_columns = label_columns
    if label_columns is not None:
      self.label_columns_indices = {name: i for i, name in
                                    enumerate(label_columns)}
    self.column_indices = {name: i for i, name in
                           enumerate(train_df.columns)}

    # Work out the window parameters.
    self.input_width = input_width
    self.label_width = label_width
    self.shift = shift

    self.total_window_size = input_width + shift

    self.input_slice = slice(0, input_width)
    self.input_indices = np.arange(self.total_window_size)[self.input_slice]

    self.label_start = self.total_window_size - self.label_width
    self.labels_slice = slice(self.label_start, None)
    self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

  def __repr__(self):
    return '\n'.join([
        f'Total window size: {self.total_window_size}',
        f'Input indices: {self.input_indices}',
        f'Label indices: {self.label_indices}',
        f'Label column name(s): {self.label_columns}'])

def split_window(self, features):
  inputs = features[:, self.input_slice, :]
  labels = features[:, self.labels_slice, :]
  if self.label_columns is not None:
    labels = tf.stack(
        [labels[:, :, self.column_indices[name]] for name in self.label_columns],
        axis=-1)

  # Slicing doesn't preserve static shape information, so set the shapes
  # manually. This way the `tf.data.Datasets` are easier to inspect.
  inputs.set_shape([None, self.input_width, None])
  labels.set_shape([None, self.label_width, None])

  return inputs, labels

def make_dataset(self, data):
  data = np.array(data, dtype=np.float32)
  ds = tf.keras.utils.timeseries_dataset_from_array(
      data=data,
      targets=None,
      sequence_length=self.total_window_size,
      sequence_stride=1,
      shuffle=True,
      batch_size=8,)

  ds = ds.map(self.split_window)

  return ds

@property
def train(self):
  return self.make_dataset(self.train_df)

@property
def val(self):
  return self.make_dataset(self.val_df)

@property
def test(self):
  return self.make_dataset(self.test_df)

@property
def example(self):
  """Get and cache an example batch of `inputs, labels` for plotting."""
  result = getattr(self, '_example', None)
  if result is None:
    # No example batch was found, so get one from the `.train` dataset
    result = next(iter(self.train))
    # And cache it for next time
    self._example = result
  return result



w2 = WindowGenerator(input_width=6, label_width=1, shift=1,
                     label_columns=['NVDA'])
w2


WindowGenerator.split_window = split_window

# Stack three slices, the length of the total window.
example_window = tf.stack([np.array(train_df[:w2.total_window_size]),
                           np.array(train_df[7:7+w2.total_window_size]),
                           np.array(train_df[12:12+w2.total_window_size])])

example_inputs, example_labels = w2.split_window(example_window)

print('All shapes are: (batch, time, features)')
print(f'Window shape: {example_window.shape}')
print(f'Inputs shape: {example_inputs.shape}')
print(f'Labels shape: {example_labels.shape}')

WindowGenerator.make_dataset = make_dataset

WindowGenerator.train = train
WindowGenerator.val = val
WindowGenerator.test = test
WindowGenerator.example = example

# Each element is an (inputs, label) pair.
w2.train.element_spec


# abalone_model = tf.keras.Sequential([
#   tf.keras.layers.Dense(64, activation='relu'),
#   tf.keras.layers.Dense(1)
# ])

# abalone_model.compile(loss = tf.keras.losses.MeanSquaredError(),
#                       optimizer = tf.keras.optimizers.Adam())

# dense = tf.keras.Sequential([
#     tf.keras.layers.Dense(units=64, activation='relu'),
#     tf.keras.layers.Dense(units=64, activation='relu'),
#     tf.keras.layers.Dense(units=1)
# ])


#modelo
lstm_model = tf.keras.Sequential([
    tf.keras.layers.LSTM(32, return_sequences=False),
    tf.keras.layers.Dense(1)
])

#compilacion
lstm_model.compile(
    loss='mean_squared_error',
    optimizer='adam',
    metrics=['mae']
)

#entrenamiento 
history = lstm_model.fit(
    w2.train,
    validation_data=w2.val,
    epochs=20
)

#evaluacion
val_loss, val_mae = lstm_model.evaluate(w2.val)
print(f"Validation MAE: {val_mae:.4f}")