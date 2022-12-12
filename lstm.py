
from keras.models import Sequential
from keras.layers import Dense, LSTM
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
from math import sqrt
from numpy import concatenate
from matplotlib import pyplot


'''
dataset=pd.read_csv("Metreology.csv")


dataset=dataset.iloc[:,2:8]

sc = MinMaxScaler(feature_range = (0, 1))
ds = sc.fit_transform(dataset)
ds=pd.DataFrame(ds)
x=ds.iloc[:,0:5]
y=ds.iloc[:,5]
'''
dataset=pd.read_csv("Pollutant_AQI.csv")


dataset=dataset.iloc[:,2:10]

sc = MinMaxScaler(feature_range = (0, 1))
ds = sc.fit_transform(dataset)
ds=pd.DataFrame(ds)
x=ds.iloc[:,0:7]
y=ds.iloc[:,7]

x=np.array(x)
y=np.array(y)

print(x.shape)

x = x.reshape(x.shape[0], 1, x.shape[1])
print(x.shape)

xtrain, xtest, ytrain, ytest=train_test_split(x, y, test_size=0.15)

model = Sequential()
model.add(LSTM(50, return_sequences = True, input_shape=(xtrain.shape[1],xtrain.shape[2])))
model.add(LSTM(20, return_sequences=True))
model.add(LSTM(10))
model.add(Dense(1))
print(model.summary())
model.compile(loss='mae', optimizer='adam')
# fit network
history = model.fit(xtrain, ytrain, epochs=50, batch_size=72, validation_data=(xtest, ytest), verbose=2, shuffle=False)
# plot history
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()

# make a prediction
yhat = model.predict(xtest)
x_test = xtest.reshape((xtest.shape[0], xtest.shape[2]))
# invert scaling for forecast
inv_yhat = concatenate((x_test, yhat), axis=1)
inv_yhat = sc.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:,-1]
# invert scaling for actual
y_test = ytest.reshape((len(ytest), 1))
inv_y = concatenate((x_test,y_test), axis=1)
inv_y = sc.inverse_transform(inv_y)
inv_y = inv_y[:,-1]
# calculate RMSE
from sklearn.metrics import mean_squared_error
rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
print('Test RMSE: %.3f' % rmse)


d=[
   [11.2, 10.1, 30.5, 1105,23.2,250, 92.7, 172],
   [10.4, 0 ,32.5, 1205, 20.1, 247.2, 97,172]
   ]
d=pd.DataFrame(d)
d=np.array(d)
print(d.shape)
#sc1=MinMaxScaler(feature_range=(0,1))
d=sc.fit_transform(d)
d=pd.DataFrame(d)
x1=d.iloc[:,:7]
x1=np.array(x1)
x1 = x1.reshape(x1.shape[0], 1, x1.shape[1])
pred=model.predict(x1)
x2=x1.reshape((x1.shape[0],x1.shape[2]))
yy=concatenate((x2,pred),axis=1)
pred_trans=sc.inverse_transform(yy)
predict=pred_trans[:-1,-1]
print("Current AQI:",predict)
