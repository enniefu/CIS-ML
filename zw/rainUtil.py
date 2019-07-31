import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from keras.utils import normalize
from scipy import stats
pd.set_option('display.max_columns', 1000)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 1000)
def offerData(url, label = "RainTomorrow", removeCol = False, location = "", removeOutliers = False, intervals = 1):
    df = pd.read_csv(url)
    # print('Size of weather data frame is :',df.shape)

    # #增加几天后下雨的真实label
    # RainTwoDay = []
    # RainThreeDay = []
    # RainFourDay = []
    # RainFiveDay = []
    # RainSixDay = []
    # RainSevenDay = []
    # Month = []
    # print(len(df['RainToday']))
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
    #
    # df['RainTwoDay'] = RainTwoDay
    # df['RainThreeDay'] = RainThreeDay
    # df['RainFourDay'] = RainFourDay
    # df['RainFiveDay'] = RainFiveDay
    # df['RainSixDay'] = RainSixDay
    # df['RainSevenDay'] = RainSevenDay
    #
    # df = df.fillna('NA')
    #
    # df.to_csv('./newWeatherAUS.csv')

    RainTwoDay = []
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

    X = df.drop('RainTomorrow', axis=1)
    y = df[label]

    X = X.values
    X = normalize(X)

    return X,y

if __name__ == '__main__':
    offerData('./weatherAUS.csv')
