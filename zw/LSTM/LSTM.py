import matplotlib.pyplot as plt
import numpy as np
from numpy import newaxis
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.layers.recurrent import LSTM
from keras.layers import Dropout
from e_util import *
from sklearn.model_selection import train_test_split
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

url = "010010 1987-2018.txt"
X,y = offer_data(url, 30, 0, 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=100)  # 分割训练集和测试集
step = 0.15
steps = 750
x = np.arange(0, steps, step)
data = X_train
SEQ_LENGTH = 29
sequence_length = SEQ_LENGTH + 1


x_train = np.array(X_train)
y_train = np.array(y_train)
x_test = np.array(X_test)
y_test = np.array(y_test)

# LSTM层的输入必须是三维的
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 13))
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 13))
print(x_train)

# Neural Network model
HIDDEN_DIM = 32
LAYER_NUM = 2
model = Sequential()
model.add(LSTM(16, input_shape=(x_train.shape[1], x_train.shape[2]), return_sequences=True))
model.add(Dropout(0.3))
model.add(LSTM(32, return_sequences=True))
model.add(Dropout(0.3))
model.add(LSTM(32, return_sequences=True))
model.add(Dropout(0.3))
model.add(LSTM(32, return_sequences=True))
model.add(Dropout(0.3))
model.add(LSTM(32,  return_sequences=True))
model.add(Dropout(0.3))
model.add(LSTM(16, return_sequences=False))
model.add(Dropout(0.3))
model.add(Dense(1))
model.add(Activation('linear'))
model.compile(loss="mse", optimizer="rmsprop", lr=0.00001)
model.summary()
BATCH_SIZE = 32
epoch = 10
model.fit(x_train, y_train, batch_size=BATCH_SIZE, verbose=1, epochs=epoch, validation_split=0.05)

# start with first frame
curr_frame = x_test[0]

# start with zeros
# curr_frame = np.zeros((100,1))

predicted = model.predict(x_test)
sum_mean = 0
for i in range(len(predicted)):  # 计算误差，这里使用的是均方根误差(Root Mean Squared Error, RMSE)
    sum_mean += (predicted[i] - y_test[i]) ** 2
sum_error = np.sqrt(sum_mean / len(y_test))
print(sum_error)
print(y_test[:10])
print(predicted[:10])
plt.figure(1)
plt.plot(predicted[:10])
plt.plot(y_test[:10])
plt.show()


