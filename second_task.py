import pandas as pd
from sklearn import linear_model
import numpy as np

read_wine = pd.read_csv('csv_file/winequality-red.csv', delimiter=';')

quality = read_wine.drop(columns=['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides',
                          'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol'],
                        axis=1, errors='ignore')
data = read_wine.drop(columns=['quality'], axis=1, errors='ignore')

data = data.to_numpy()
quality = quality['quality'].to_numpy()

split_size = int(0.8 * len(data))
data_train, quality_train = data[:split_size], quality[:split_size]
data_test, quality_test = data[split_size:], quality[split_size:]

print(data_train)

reg = linear_model.LinearRegression()
reg.fit(data_train, quality_train)

summ = 0
for x in range(len(quality_test)):
    prd = reg.predict([data_test[x]])
    if np.rint(prd) == quality_test[x]:
      summ += 1

print(summ)
print(len(quality_test))
print((summ/len(quality_test)) * 100)
