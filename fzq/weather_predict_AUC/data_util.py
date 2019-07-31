import numpy as np
import pandas as pd

'''

df = pd.read_csv(r"D:/weatherAUS.csv")

pd.set_option('display.max_columns',1000)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth',1000)
#
# print(df.corr() )

# #替换
# df['RainToday'].replace({'No': 0, 'Yes': 1},inplace = True)
# df['RainTomorrow'].replace({'No': 0, 'Yes': 1},inplace = True)


#清洗数据
# print(df.head())
df = df.dropna()

#替换
df['RainToday'].replace({'No': 0, 'Yes': 1},inplace = True)
df['RainTomorrow'].replace({'No': 0, 'Yes': 1},inplace = True)

#取出独特的值
# print(df['Location'].unique())

categorical_columns = ['WindGustDir', 'WindDir3pm', 'WindDir9am']
for col in categorical_columns:
    print(np.unique(df[col]))

df = pd.get_dummies(df, columns=categorical_columns)

#展示第4到9行
# print(df.iloc[4:9])

print(df.values)

'''
def read_data(url):
    df = pd.read_csv(url)

    #丢弃所有的缺失值
    # df = df.dropna()

    #将文字信息替换为1\0
    df['RainToday'].replace({'No': 0, 'Yes': 1},inplace = True)
    df['RainTomorrow'].replace({'No': 0, 'Yes': 1},inplace = True)

    #one-hot编码
    categorical_columns = ['WindGustDir', 'WindDir3pm', 'WindDir9am','Location']
    df = pd.get_dummies(df, columns=categorical_columns)

    #读取日期信息并添加one-hot月份的特征     一共12个

    #删除日期信息
    df.drop(['Date','RainTomorrow'], axis=1, inplace=True)



    return df


def offer_one_day_data(url):
    df = read_data(url)

    #获取X，y
    y = df['RainTomorrow']
    X = df.drop(['RainTomorrow','Date'], axis=1)

    return X,y

def offer_rainfall_data(url,X_length,interval,y_length):
    texts = read_data(url)
    texts  =texts.fillna('两开花')

    texts = texts.values

    X = []
    y = []



    # 试图使用7天之前的数据来预测平均温度
    for i in range(len(texts)):

        X_tmp = []
        y_tmp = []
        if i + y_length + interval< len(texts) and i+interval+y_length+X_length <len(texts) :
            for j in range(i,i+X_length):
                X_tmp.append(texts[j])
            for j in range(i+interval+X_length,i+interval+y_length+X_length):

                #取自己想要的数据
                y_tmp.append(texts[j][2])
            X.append(X_tmp)
            y.append(y_tmp)


    X_pro = []
    y_pro = []


    for i in range(len(X)):
        flag = 0
        for day_data in X[i]:
            for x in day_data:
                if x == '两开花':
                    flag=1
                    break

        for day_data in y[i]:
            if isinstance(day_data,str):
                if day_data == '两开花':
                    flag=1
                    break

        if flag == 0:
            X_pro.append(X[i])
            y_pro.append(y[i])



    return X_pro,y_pro






def offer_date_whether_rain(X,y):

    y = np.array(y,dtype=float)

    ans = np.nonzero(y)
    y_pro = np.zeros(y.shape)

    for i,j in zip(ans[0],ans[1]):
        y_pro[i][j] = 1

    return X,y_pro



if __name__ == '__main__':
    pd.set_option('display.max_columns', 1000)
    pd.set_option('display.width', 1000)
    pd.set_option('display.max_colwidth', 1000)

    url = r"D:/weatherAUS.csv"
    df = read_data(url)
    print(df)



    X,y = offer_rainfall_data(url,3,20,5)
    print(X[0])
    print(y[0])
    #
    X, y = offer_date_whether_rain(X, y)
    print(y[0])

