import keras
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import rainUtil
import matplotlib.pyplot as plt
import graphviz
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import BaggingClassifier

labels = ['0', '1']

def plot_confusion_matrix(cm, title='Confusion Matrix', cmap=plt.cm.binary):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    xlocations = np.array(range(len(labels)))
    plt.xticks(xlocations, labels, rotation=90)
    plt.yticks(xlocations, labels)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

from sklearn import tree

# model = keras.models.Sequential([
#
#     keras.layers.Dense(128, input_shape=(23,), activation='relu'),
#     keras.layers.BatchNormalization(),
#     keras.layers.Dropout(0.5),
#
#     keras.layers.Dense(128, activation='relu'),
#     keras.layers.BatchNormalization(),
#     keras.layers.Dropout(0.5),
#
#     keras.layers.Dense(128, activation='relu'),
#     keras.layers.BatchNormalization(),
#     keras.layers.Dropout(0.5),
#
#     keras.layers.Dense(128, activation='relu'),
#     keras.layers.BatchNormalization(),
#     keras.layers.Dropout(0.5),
#
#     keras.layers.Dense(125, activation='relu'),
#     keras.layers.BatchNormalization(),
#     keras.layers.Dropout(0.5),
#
#     keras.layers.Dense(1, activation='sigmoid')
# ])
#


X, y = rainUtil.offerData("./weatherAUS.csv", "RainTomorrow", selectK=3)
print(X)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.2)
X_val, X_test, y_val, y_test = train_test_split(X_val, y_val, test_size = 0.5)
# model.compile(loss='binary_crossentropy',
#               optimizer=Adam(0.00001),
#               metrics=['acc'])
#
#
# model.fit(X_train, y_train,
#                     epochs=150,
#                     validation_data=(X_val, y_val),
#                     verbose=1,
#                    )

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
import time

import numpy as np

predictTimes = []

t0=time.time()
clf_logreg = LogisticRegression(random_state=0)
clf_logreg.fit(X_train,y_train)
y_pred = clf_logreg.predict(X_test)
score = accuracy_score(y_test,y_pred)
print('LRAccuracy :',score)
print('Time taken :' , time.time()-t0)


# cm = confusion_matrix(y_test, y_pred)
# np.set_printoptions(precision=2)
# cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
# plot_confusion_matrix(cm_normalized, title='Logistic Regression normalized confusion matrix')
# plt.show()

predictTimes.append(np.transpose(y_pred).tolist())

clf_rf = RandomForestClassifier()
clf_rf.fit(X_train,y_train)

y_pred = clf_rf.predict(X_test)
score = accuracy_score(y_test, y_pred)
print('RFAccuracy :',score)
print('Time taken :' , time.time()-t0)




predictTimes.append(np.transpose(y_pred).tolist())

t0=time.time()
clf_dt = DecisionTreeClassifier(random_state=0)
clf_dt.fit(X_train,y_train)

y_pred = clf_dt.predict(X_test)
score = accuracy_score(y_test,y_pred)
print('DTAccuracy :',score)
print('Time taken :' , time.time()-t0)




predictTimes.append(np.transpose(y_pred).tolist())

t0=time.time()
mlp = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(5, 5), random_state=1)
mlp.fit(X_train,y_train)
y_pred = mlp.predict(X_test)
score = accuracy_score(y_test,y_pred)
print('NNAccuracy :',score)
print('Time taken :' , time.time()-t0)

predictTimes.append(np.transpose(y_pred).tolist())



t0=time.time()
mnb = MultinomialNB()   # 使用默认配置初始化朴素贝叶斯
mnb.fit(X_train,y_train)    # 利用训练数据对模型参数进行估计
y_pred = mnb.predict(X_test)     # 对参数进行预测
score = accuracy_score(y_test, y_pred)
print('NNAccuracy :',score)
print('Time taken :' , time.time()-t0)
print(predictTimes)

Bag = BaggingClassifier()
Bag.fit(X_train, y_train)
y_pred = Bag.predict(X_test)
score = accuracy_score(y_test, y_pred)
print('BagAccuracy :', score)
print('Time taken :', time.time()-t0)
print(predictTimes)

# clf_rf = RandomForestClassifier()
# clf_rf.fit(predictTimes, y_test)
#
# y_pred = clf_rf.predict(X_test)
# score = accuracy_score(y_test, y_pred)
# print('RFAccuracy :', score)


# scores = []
# scores.append(accuracy_score(y_today,y))
# scores.append(accuracy_score(y_today,y_RainTwoDay))
# scores.append(accuracy_score(y_today,y_RainThreeDay))
# scores.append(accuracy_score(y_today,y_RainFourDay))
# scores.append(accuracy_score(y_today,y_RainFiveDay))
# scores.append(accuracy_score(y_today,y_RainSixDay))
# scores.append(accuracy_score(y_today,y_RainSevenDay))

# plt.plot(scores)
# plt.show()

# t0=time.time()
# clf_svc = svm.SVC(kernel='linear')
# clf_svc.fit(X_train,y_train)
# y_pred = clf_svc.predict(X_test)
# score = accuracy_score(y_test,y_pred)
# print('Accuracy :',score)
# print('Time taken :' , time.time()-t0)