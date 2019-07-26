import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from e_util import offer_data



X,y = offer_data(r"D:\010010 1987-2018.txt",14,0,365)

X = np.array(X)
y = np.array(y)

X=np.reshape(X,(X.shape[0],-1))
y=np.reshape(y,(y.shape[0],-1))
print(X.shape)
print(y.shape)


scaler = StandardScaler(copy=True, with_mean=True, with_std=True) # 标准化转换
scaler.fit(X)  # 训练标准化对象
X = scaler.transform(X)   # 转换数据集

X_train, X_test, y_train, y_test = train_test_split(X,y)

model = MLPRegressor(hidden_layer_sizes=(30,10),alpha=0.1,solver='lbfgs')


model.fit(X_train,y_train)
y_pred = model.predict(X_test)

print(y_pred.shape)

day = [x for x in range(20)]


plt.axis()

plt.plot(day,y_pred[:20],c='purple',label='predict_value')
plt.plot(day,y_test[:20],c='red',label=u'true_value')
plt.legend(fontsize=15,loc='best')
plt.show()

#计算差值
RMSE_loss = np.sqrt(np.sum(np.square(y_test-y_pred))/len(y_pred))
print(RMSE_loss)


