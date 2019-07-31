import numpy as np

y = np.array([[1,1.1,0.0],[0,0,0]])
ans = np.nonzero(y)
y_pro = np.zeros(np.array(y).shape)

for i, j in zip(ans[0], ans[1]):
    y_pro[i][j] = 1

print(y_pro)