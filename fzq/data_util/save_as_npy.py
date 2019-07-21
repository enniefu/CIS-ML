import numpy as np
import readOp

url=r"D:\010010 1987-2018.txt"
texts=readOp.readOpcal(url)

X=[]
y=[]

#试图使用7天之前的数据来预测平均温度
for i in range(len(texts)):
    X_tmp = []
    if i+7<len(texts) and texts[i+7]:
        y.append(texts[i+7][0])
        for j in range(i,i+7):
            X_tmp.append(texts[j])
        X.append(X_tmp)

X=np.array(X)
y=np.array(y)

np.save(r'X.npy',X)
np.save(r'y.npy',y)

X_load=np.load('X.npy')
y_load=np.load('y.npy')

# print(X_load)
# print(len(X_load))
