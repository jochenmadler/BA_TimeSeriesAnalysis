import pandas as pd
from sklearn.metrics import mean_squared_error
from math import sqrt
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib import pyplot as plt
plt.rcParams["figure.figsize"] = (10,4)

mydateparser = lambda x: pd.datetime.strptime(str(x), '%Y-%m-%d %H:%M:%S')
btc_series = pd.read_excel('Data/BTC_closing.xlsx',squeeze=True, parse_dates=[0], index_col=0, date_parser=mydateparser)

X = btc_series.values
train_size = int(len(X) * 0.5)
train, test = X[0:train_size], X[train_size:]
#walk-forward validation
history = [x for x in train]
predictions = list()
for i in range(len(test)):
    #prediction
    yhat = history[-1]
    predictions.append(yhat)
    #observation
    obs = test[i]
    history.append(obs)
#report performance (rmse)
mse = mean_squared_error(test, predictions)
rmse = sqrt(mse)

print('RMSE: %.3f' % rmse)
plt.plot(test, color = 'blue', linewidth = 1)
plt.plot(predictions, color = 'orange', linewidth = 1)
