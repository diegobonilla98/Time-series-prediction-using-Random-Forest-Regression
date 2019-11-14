import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import scale
from sklearn.ensemble import RandomForestRegressor


data = pd.read_csv('weatherHistory.csv')

temperatures = data['Temperature (C)'].values
temperatures = scale(temperatures)

last_days = []
today = []
step = 1
max_len = 5
for idx in range(0, len(temperatures) - max_len, step):
    last_days.append(temperatures[idx: idx + max_len])
    today.append(temperatures[idx + max_len])

x = np.array(last_days)
y = np.array(today)

split = x.shape[0] // 5
X_train = x[:-split]
X_test = x[-split:]
y_train = y[:-split]
y_test = y[-split:]

print(X_train.shape, X_test.shape)
print(y_train.shape, y_test.shape)


# probar scale, standardScaler y sin nada

regr = RandomForestRegressor(random_state=42, n_estimators=50)

regr.fit(X_train, y_train)
y_pred = regr.predict(X_test)

ant = X_test.shape[0] + 5000
x_axis = np.arange(ant) + len(temperatures) - ant
x_axis_nxt = np.arange(len(y_pred)) + y_train.shape[0]
plt.plot(x_axis, temperatures[-ant:], 'k', linewidth=2)
plt.plot(x_axis_nxt, y_pred, 'r', linewidth=1)
plt.show()

