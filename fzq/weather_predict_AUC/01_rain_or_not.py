from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import matplotlib.pyplot as plt
import graphviz
from sklearn import tree

from data_util import offer_date_whether_rain,offer_rainfall_data

def random_forest(url,X_length,interval,y_length):
    #获取数据
    X,y = offer_rainfall_data(url,X_length,interval,y_length)
    X,y = offer_date_whether_rain(X,y)

    #reshape，处理成sklearn能识别的格式
    X = np.array(X)
    y = np.array(y)

    X = np.reshape(X,(X.shape[0],-1))
    y = y.ravel()

    # print(X.shape)
    # print(y.shape)

    #标准化转换
    scaler=StandardScaler()
    scaler.fit(X)
    X = scaler.fit_transform(X)

    #拆分数据
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 42)

    #获取模型
    model = RandomForestClassifier(n_estimators=100)

    #训练模型
    model.fit(X_train,y_train)
    prediction = model.predict(X_test)

    #预测数据
    prediction = np.array(prediction)
    y_test = np.array(y_test)

    acc = np.mean(y_test==prediction)
    return acc
    # print(y_test[0])
    # print(prediction[0])
    # print(y_test[1])
    # print(prediction[1])
    # print(y_test[2])
    # print(prediction[2])
    # for i in range(20):
    #     print(y_test[i])

def deccision_tree(url,X_length,interval,y_length):
    #获取数据
    X,y = offer_rainfall_data(url,X_length,interval,y_length)
    X,y = offer_date_whether_rain(X,y)

    #reshape，处理成sklearn能识别的格式
    X = np.array(X)
    y = np.array(y)

    X = np.reshape(X,(X.shape[0],-1))
    y = y.ravel()

    # print(X.shape)
    # print(y.shape)

    #标准化转换
    scaler=StandardScaler()
    scaler.fit(X)
    X = scaler.fit_transform(X)

    #拆分数据
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 42)

    #获取模型
    model = DecisionTreeClassifier()

    #训练模型
    model.fit(X_train,y_train)

    dot_data = tree.export_graphviz(model, out_file=None)  # doctest: +SKIP
    graph = graphviz.Source(dot_data)  # doctest: +SKIP
    # 在同级目录下生成tree.pdf文件
    graph.render("tree")  # doctest: +SKIP


    prediction = model.predict(X_test)

    #预测数据
    prediction = np.array(prediction)
    y_test = np.array(y_test)

    acc = np.mean(y_test==prediction)
    return acc





if __name__ == '__main__':
    url = r"D:/weatherAUS.csv"
    acc_list = []
    day_length = 7
    for i in range(day_length):
        print(i)
        acc = random_forest(url,1,i,1)
        acc_list.append(acc)


    shift_acc = [0.762,0.708,0.693,0.690,0.685,0.688,0.687]

    plt.plot([x+1 for x in range(day_length)],acc_list,label='predict')
    plt.plot([x + 1 for x in range(day_length)], shift_acc,label='baseline' )
    plt.legend()
    plt.show()


    # acc = random_forest(url,1,0,1)
    # print(acc)



    #连续预测7天是否下雨
