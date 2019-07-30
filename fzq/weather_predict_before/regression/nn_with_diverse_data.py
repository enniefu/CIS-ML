import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from e_util import offer_data



X,y = offer_data(r"D:\010010 1987-2018.txt",30,0,7)

X = np.array(X)
y = np.array(y)

X=np.reshape(X,(X.shape[0],-1))
y=np.reshape(y,(y.shape[0],-1))
print(X.shape)
print(y.shape)

scaler = StandardScaler(copy=True, with_mean=True, with_std=True) # 标准化转换
# scaler.fit(X)  # 训练标准化对象
# X = scaler.transform(X)   # 转换数据集
# X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=0)

'''
避免overlap
'''
X_train = X[:7000]
y_train = y[:7000]

X_test = X[7000:]
y_test = y[7000:]



model = MLPRegressor(hidden_layer_sizes=(30,10),alpha=0.001,max_iter=2000)


model.fit(X_train,y_train)
y_pred = model.predict(X_test)


# print(y_pred.shape)
# exit()

day = [x for x in range(7)]

for i in range(len(y_pred)):
    plt.figure(i)
    plt.axis()
    
    plt.plot(day,y_pred[i],c='purple',label='predict_value')
    plt.plot(day,y_test[i],c='red',label='true_value')
    plt.legend(fontsize=15,loc='best')
    plt.show()

#计算差值
RMSE_loss = np.sqrt(np.sum(np.square(y_test-y_pred))/(y_pred.shape[0]*y_pred.shape[1]))
print(RMSE_loss)



