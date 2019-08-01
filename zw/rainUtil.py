import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from keras.utils import normalize
from scipy import stats
from sklearn.feature_selection import SelectKBest, chi2
from sklearn import preprocessing
import seaborn as sns
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', 1000)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 1000)
def offerData(url, label = "RainTomorrow", removeCol = False, location = "", removeOutliers = False, intervals = 1, selectK = 1):
    df = pd.read_csv(url)
    print('Size of weather data frame is :',df.shape)

    # countYes = 0
    # countNo = 0
    # for i in range(len(df['RainTomorrow'])):
    #     if (df['RainTomorrow'][i] == 'Yes') :
    #         countYes = countYes + 1
    #     else:
    #         countNo = countNo + 1
    #
    # print(countYes)
    # print(countNo)

    #增加几天后下雨的真实label
    # RainTwoDay = []
    # RainThreeDay = []
    # RainFourDay = []
    # RainFiveDay = []
    # RainSixDay = []
    # RainSevenDay = []
    # for i in range(len(df['RainToday'])):
    #     if i + 1 >= len(df['RainToday']):
    #         RainTwoDay.append(np.nan)
    #         RainThreeDay.append(np.nan)
    #         RainFourDay.append(np.nan)
    #         RainFiveDay.append(np.nan)
    #         RainSixDay.append(np.nan)
    #         RainSevenDay.append(np.nan)
    #         continue
    #     if df['Location'][i] == df['Location'][i + 1]:
    #         RainTwoDay.append(df['RainTomorrow'][i + 1])
    #     else:
    #         RainTwoDay.append(np.nan)
    #     if i + 2 >= len(df['RainToday']):
    #         RainThreeDay.append(np.nan)
    #         RainFourDay.append(np.nan)
    #         RainFiveDay.append(np.nan)
    #         RainSixDay.append(np.nan)
    #         RainSevenDay.append(np.nan)
    #         continue
    #     if df['Location'][i] == df['Location'][i + 2]:
    #         RainThreeDay.append(df['RainTomorrow'][i + 2])
    #     else:
    #         RainThreeDay.append(np.nan)
    #     if i + 3 >= len(df['RainToday']):
    #         RainFourDay.append(np.nan)
    #         RainFiveDay.append(np.nan)
    #         RainSixDay.append(np.nan)
    #         RainSevenDay.append(np.nan)
    #         continue
    #     if df['Location'][i] == df['Location'][i + 3]:
    #         RainFourDay.append(df['RainTomorrow'][i + 3])
    #     else:
    #         RainFourDay.append(np.nan)
    #     if i + 4 >= len(df['RainToday']):
    #         RainFiveDay.append(np.nan)
    #         RainSixDay.append(np.nan)
    #         RainSevenDay.append(np.nan)
    #         continue
    #     if df['Location'][i] == df['Location'][i + 4]:
    #         RainFiveDay.append(df['RainTomorrow'][i + 4])
    #     else:
    #         RainFiveDay.append(np.nan)
    #     if i + 5 >= len(df['RainToday']):
    #         RainSixDay.append(np.nan)
    #         RainSevenDay.append(np.nan)
    #         continue
    #     if df['Location'][i] == df['Location'][i + 5]:
    #         RainSixDay.append(df['RainTomorrow'][i + 5])
    #     else:
    #         RainSixDay.append(np.nan)
    #     if i + 6 >= len(df['RainToday']):
    #         RainSevenDay.append(np.nan)
    #         continue
    #     if df['Location'][i] == df['Location'][i + 6]:
    #         RainSevenDay.append(df['RainTomorrow'][i + 6])
    #     else:
    #         RainSevenDay.append(np.nan)
    #
    # df['RainTwoDay'] = RainTwoDay
    # df['RainThreeDay'] = RainThreeDay
    # df['RainFourDay'] = RainFourDay
    # df['RainFiveDay'] = RainFiveDay
    # df['RainSixDay'] = RainSixDay
    # df['RainSevenDay'] = RainSevenDay

    #增加一个Month的label
    Month = []

    for i in range(len(df['Date'])):
        Month.append(int(df['Date'][i][5:7]))

    df['Month'] = Month

    df = df.dropna(how='any')

    print('Size of weather data frame is :', df.shape)

    # df.to_csv('./newWeatherAUS.csv')
    # if (intervals > 1):
    #     for i in range(len(texts)):
    #         X_tmp = []
    #         y_tmp = []
    #         if i + y_length + interval< len(texts) and i+interval+y_length+X_length <len(texts) :
    #             for j in range(i,i+X_length):
    #                 X_tmp.append(texts[j])
    #             for j in range(i+interval+X_length,i+interval+y_length+X_length):
    #                 y_tmp.append(texts[j][0])
    #             X.append(X_tmp)
    #             y.append(y_tmp)

    df = df.drop(columns=['Date'], axis=1)
    if removeCol:
        df = df.drop(columns=['Sunshine','Evaporation','Cloud3pm','Cloud9am','Location','RISK_MM','Date'],axis=1)
        categorical_columns = ['WindGustDir', 'WindDir3pm', 'WindDir9am']
    else:
        categorical_columns = ['WindGustDir', 'WindDir3pm', 'WindDir9am', 'Location']

    df = df.dropna(how='any')

    df['RainToday'].replace({'No': 0, 'Yes': 1}, inplace=True)
    df['RainTomorrow'].replace({'No': 0, 'Yes': 1}, inplace=True)

    # df['RainTwoDay'].replace({'No': 0, 'Yes': 1}, inplace=True)
    # df['RainThreeDay'].replace({'No': 0, 'Yes': 1}, inplace=True)
    # df['RainFourDay'].replace({'No': 0, 'Yes': 1}, inplace=True)
    # df['RainFiveDay'].replace({'No': 0, 'Yes': 1}, inplace=True)
    # df['RainSixDay'].replace({'No': 0, 'Yes': 1}, inplace=True)
    # df['RainSevenDay'].replace({'No': 0, 'Yes': 1}, inplace=True)

    if location != "":
        df = df


    # 非one-hot编码
    # df['Date'] = df['Date'].astype('category').cat.codes
    df['Location'] = df['Location'].astype('category').cat.codes
    df['WindGustDir'] = df['WindGustDir'].astype('category').cat.codes
    df['WindDir9am'] = df['WindDir9am'].astype('category').cat.codes
    df['WindDir3pm'] = df['WindDir3pm'].astype('category').cat.codes

    # one-hot
    # df = pd.get_dummies(df, columns=categorical_columns)

    if removeOutliers:
        z = np.abs(stats.zscore(df._get_numeric_data()))
        df = df[(z < 3).all(axis=1)]


    df.reset_index(drop=True, inplace=True)

    # y_RainToday = df['RainToday']
    # y_RainTwoDay = df['RainTwoDay']
    # y_RainThreeDay = df['RainThreeDay']
    # y_RainFourDay = df['RainFourDay']
    # y_RainFiveDay = df['RainFiveDay']
    # y_RainSixDay = df['RainSixDay']
    # y_RainSevenDay = df['RainSevenDay']

    # scaler = preprocessing.MinMaxScaler()
    # scaler.fit(df)
    # df = pd.DataFrame(scaler.transform(df), index=df.index, columns=df.columns)

    X = df.drop(['RainTomorrow'], axis=1)

    # X = df.drop(['RainTomorrow', 'RainThreeDay', 'RainTwoDay', 'RainFourDay', 'RainFiveDay', 'RainSixDay',
    #              'RainSevenDay'], axis=1)

    X = X.loc[:, ]
    X = df["Rainfall"]
    y = df[label]
    # selector = SelectKBest(chi2, k=selectK)
    # selector.fit(X, y)
    # print(selector)
    # X_new = selector.transform(X)
    # print(X.columns[selector.get_support(indices=True)])  # top 3 columns

    return X, y#, y_RainToday, y_RainTwoDay, y_RainThreeDay, y_RainFourDay, y_RainFiveDay, y_RainSixDay, y_RainSevenDay

if __name__ == '__main__':
    offerData('./weatherAUS.csv')
