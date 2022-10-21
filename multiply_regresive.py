from sklearn import linear_model
import numpy as np


arr_x = np.array([[18, 71, 69, 60],
                 [27, 74, 97, 27],
                 [5, 42, 33, 41],
                 [49, 38, 44, 56],
                 [83, 76, 42, 37],
                 [91, 81, 34, 42],
                 [78, 93, 35, 49]])

arr_y = np.array([77, 337, -27, 123, 196, 155, 76])

arr_cef = np.array([97, 41, 89, 93])

print(arr_x)
reg = linear_model.LinearRegression()
reg.fit(arr_x, arr_y)
print(reg.predict([arr_cef]))