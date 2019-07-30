import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from e_util import *

X = np.load(r"D:\AA_Programming\pycharm\CIS-ML\fzq\data_util\X.npy")
y = np.load(r"D:\AA_Programming\pycharm\CIS-ML\fzq\data_util\y.npy")

X=np.reshape(X,(X.shape[0],-1))


scaler = StandardScaler() # 标准化转换
scaler.fit(X)  # 训练标准化对象
X = scaler.transform(X)   # 转换数据集

X_train, X_test, y_train, y_test = train_test_split(X,y)

model = MLPRegressor(hidden_layer_sizes=(100,100))
model.fit(X_train,y_train)
y_pred = model.predict(X_test)

day = [x for x in range(20)]


plt.axis()

plt.plot(day,y_pred[:20],c='purple',label='predict_value')
plt.plot(day,y_test[:20],c='red',label=u'true_value')
plt.legend(fontsize=15,loc='best')
plt.show()

