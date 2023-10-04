import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.layers import Input, Dense
from keras.models import Model, Sequential
import numpy as np
import matplotlib.pyplot as plt
import h5py
import pandas as pd
import math

file_path = '../scripts/evaluate_ttbar.h5'

f = h5py.File(file_path, 'r')
raw_data = np.array(f['data'])
pid = np.array(f['pid'])
x_back_arr = []
x_sig_arr = []
for i in range(len(pid)):
    if pid[i] == 0.:
        x_back_arr.append(raw_data[i].ravel())
    else:
        x_sig_arr.append(raw_data[i].ravel())
x_sig = np.array(x_sig_arr)
x_back = np.array(x_back_arr)

x_vals = np.concatenate([x_sig,x_back])
y_vals = np.concatenate([np.ones(len(x_sig)),np.zeros(len(x_back))])
X_train, X_val, Y_train, Y_val = train_test_split(x_vals, y_vals, test_size=0.5)

model = Sequential()
model.add(Dense(128, input_dim=700, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

myhistory = model.fit(x_vals, y_vals, epochs=20,validation_data=(X_val, Y_val),batch_size=1024)

from sklearn.metrics import roc_curve, auc

preds = model.predict(X_val,batch_size=1000)

fpr, tpr, _ = roc_curve(Y_val, preds)

plt.plot(tpr,1.-fpr)
plt.savefig('roc curve')