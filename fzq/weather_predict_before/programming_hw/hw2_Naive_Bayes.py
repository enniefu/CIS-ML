from sklearn.metrics import precision_score, recall_score, f1_score
from math import log
from hw_util import offer_data, getDiscreteProbability,Get_likelihood,compute_mean_and_variance,Get_mean_and_variance
from hw_util import offer_test_data
import numpy as np

X_train,y_train = offer_data("adult.data")
X_test,y_test = offer_data("adult.test")

#计算先验概率   P(y=0)   P(y=1)
p1 = y_train.count(1)/len(y_train)
p0 = y_train.count(0)/len(y_train)

#计算类条件概率
Y_probability_0, Y_probability_1 = getDiscreteProbability(X_train, y_train)

mean_and_var = Get_mean_and_variance(X_train,y_train)


#使用先验概率和类条件概率进行分类：
y_pred = []

for person in X_test:
    score_list = []
    for c_label in (0,1):

        if c_label == 0:

            score = log(p0)
            for i  in range(len(person)):
                # 处理离散数据
                if i not in (0,2,4,10,11,12) and person[i]!='?' :
                    try:
                        score = score +log(Y_probability_0[person[i]])
                    except:
                        print(person)
                        print(person[i])
                        print(Y_probability_1[person[i]])
                #处理连续数据
                elif person[i]!='?':
                    score += log(Get_likelihood(mean_and_var, i, 0, person[i]))
            score_list.append(score)

        elif c_label == 1:

            score = log(p1)

            for i in range(len(person)):
                # 处理离散数据
                if i not in (0, 2, 4, 10, 11, 12) and person[i]!='?':
                    try:
                        score = score + log(Y_probability_1[person[i]])
                    except:
                        print(person)
                        print(person[i])
                        print(Y_probability_1[person[i]])

                # 处理连续数据
                elif person[i]!='?':
                    score += log(Get_likelihood(mean_and_var, i, 1, person[i]))
            score_list.append(score)

    lable = score_list.index(max(score_list))
    y_pred.append(lable)


y_pred = np.array(y_pred)
y_test = np.array(y_test)


#计算准确率
accuracy = np.sum(y_pred==y_test)/len(y_pred)
precision = precision_score(y_pred, y_test)
recall = recall_score(y_pred, y_test)
f1score = f1_score(y_pred, y_test)

print("acc={},precision={},recall={},f1score={}".format(accuracy,precision,recall,f1score))




