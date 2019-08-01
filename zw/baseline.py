import rainUtil
from sklearn.metrics import mean_squared_error
import numpy as np
data, y = rainUtil.offerData('./weatherAUS.csv')
print(np.sqrt(mean_squared_error(data,y)))
