import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import math,pylab,matplotlib,numpy



def offer_test_data(url):
    X,y = offer_data(url)

    X_pro = []
    y_pro = []

    flag = 0
    for i in range(len(X)):
        for j in range(len(X[0])):
            if i == '?':
                flag = 1
                break
        if flag == 0:
            X_pro.append(X[i])
            y_pro.append(y[i])
        flag = 0
    return X_pro,y_pro



def read_data(url):
    '''
    输入原始数据，清洗数据并输出
    :param url:
    :return:
    '''
    with open(url,'r') as f:
        data = []
        while True:
            lines = f.readline()  # 整行读取数据
            if not lines or lines =='':
                break

            #处理一行的数据
            dataline=[]
            texts = lines.split(',')
            for text in texts:
                text = text.strip()
                if text != '':
                    dataline.append(text)

            if dataline:
                data.append(dataline)

        return data

def offer_data(url):
    data = read_data(url)

    X = []
    y = []

    width = len(data[0])

    for line in data:
        line_data =[]
        for j in range(width):
            if j in (0,2,4,10,11,12):
                line_data.append(float(line[j]))
            elif j!= 14:
                line_data.append(line[j])

            else:

                if line[j] == '>50K.' or line[j] == '>50K':
                    y.append(1)

                else:
                    y.append(0)

        X.append(line_data)
    return X,y
def probabilityValueInNeed(data, label, index, value, need):
    count_need = 0
    count_value = 0
    for i in range(len(data)):
        if label[i] == need:
            count_need = count_need + 1
            if data[i][index] == '?':
                count_need = count_need - 1
            if data[i][index] == value:
                count_value = count_value + 1
    if count_value == 0:
        count_value = count_value + 1
        count_need = count_value + 2
    return count_value / count_need


def getDiscreteProbability(X, y):
    __DICT__ = [1, 3, 5, 6, 7, 8, 9, 13]

    DICTIONARY = {}
    DICTIONARY[0] = {0: 'Private', 1: 'Self-emp-not-inc', 2: 'Self-emp-inc', 3: 'Federal-gov', 4: "Local-gov",
                     5: 'State-gov', 6: 'Without-pay', 7: 'Never-worked'}
    DICTIONARY[1] = {0: 'Bachelors', 1: 'Some-college', 2: '11th', 3: 'HS-grad', 4: 'Prof-school',
                     5: 'Assoc-acdm', 6: 'Assoc-voc', 7: '9th', 8: '7th-8th', 9: '12th', 10: 'Masters',
                     11: '1st-4th', 12: '10th', 13: 'Doctorate', 14: '5th-6th', 15: 'Preschool'}
    DICTIONARY[2] = {0: 'Married-civ-spouse', 1: 'Divorced', 2: "Never-married", 3: "Separated", 4: "Widowed",
                     5: "Married-spouse-absent", 6: "Married-AF-spouse"}
    DICTIONARY[3] = {0: 'Tech-support', 1: 'Craft-repair', 2: "Other-service", 3: "Sales", 4: "Exec-managerial",
                     5: "Prof-specialty", 6: "Handlers-cleaners", 7: 'Machine-op-inspct', 8: "Adm-clerical",
                     9: "Farming-fishing", 10: "Transport-moving", 11: "Priv-house-serv", 12: "Protective-serv",
                     13: "Armed-Forces"}
    DICTIONARY[4] = {0: 'Wife', 1: "Own-child", 2: "Husband", 3: "Not-in-family", 4: "Other-relative",
                     5: "Unmarried"}
    DICTIONARY[5] = {0: 'White', 1: "Asian-Pac-Islander", 2: "Amer-Indian-Eskimo", 3: "Other", 4: "Black"}
    DICTIONARY[6] = {0: 'Female', 1: "Male"}
    DICTIONARY[7] = {0: 'United-States', 1: "Cambodia", 2: "England", 3: "Puerto-Rico", 4: "Canada",
                     5: "Germany", 6: 'Outlying-US(Guam-USVI-etc)', 7: 'India', 8: 'Japan', 9: 'Greece',
                     10: 'South', 11: 'China', 12: 'Cuba', 13: 'Iran', 14: 'Honduras', 15: 'Philippines',
                     16: 'Italy', 17: 'Poland', 18: 'Jamaica', 19: 'Vietnam', 20: 'Mexico', 21: 'Portugal',
                     22: 'Ireland', 23: 'France', 24: 'Dominican-Republic', 25: 'Laos', 26: 'Ecuador', 27: 'Taiwan',
                     28: 'Haiti', 29: 'Columbia', 30: 'Hungary', 31: 'Guatemala', 32: 'Nicaragua', 33: 'Scotland',
                     34: 'Thailand',
                     35: 'Yugoslavia', 36: 'El-Salvador', 37: 'Trinadad&Tobago', 38: 'Peru', 39: 'Hong',
                     40: 'Holand-Netherlands'}
    answer = []
    countY = {}
    countY[0] = 0
    countY[1] = 0
    for i in range(2):
        for j in range(len(y)):
            if (y[j] == i):
                countY[i] = countY[i] + 1
    Y_probability_1 = {}
    Y_probability_0 = {}

    for i in range(8):
        for k in range(len(DICTIONARY[i])):
            now = DICTIONARY[i][k]
            count = 0
            Y_probability_0[now] = probabilityValueInNeed(X, y, __DICT__[i], now, 0)
            Y_probability_1[now] = probabilityValueInNeed(X, y, __DICT__[i], now, 1)

    return Y_probability_0, Y_probability_1


def compute_mean_and_variance(data, col, label, y):
    # y 是所属类别的list
    # label是类别的值，0或1
    # 遍历将‘？’“去除”。先确定对应label的数据，再数？的个数，得到有效数据的条数，再计算均值和方差
    count = 0
    for i0 in range(len(data)):
        for j0 in range(len(data[i0])):
            if y[i0] == label and j0 == col:
                if data[i0][j0] == '?':
                    count = count + 1
    length = len(data) - count  # 去掉？后 given y 的长度

    sum = 0
    for i in range(len(data)):
        for j in range(len(data[i])):
            if y[i] == label and j == col:
                if data[i][j] == '?':
                    continue
                else:
                    sum += data[i][j]
    mean = sum/length

    sum2 = 0
    for i in range(len(data)):
        for j in range(len(data[i])):
            if y[i] == label and j == col:
                if data[i][j] == '?':
                    continue
                else:
                    sum2 += pow(data[i][j], 2)
    variance = sum2/length

    return mean, variance


def Get_mean_and_variance(data, y):
    result = numpy.zeros([2, 14,2])
    for i in range(2):
        for j in range(14):
            if j == 0 or j == 2 or j == 4 or j == 10 or j == 11 or j == 12:
                result[i][j][0], result[i][j][1] = compute_mean_and_variance(data, j, i, y)
            else:
                result[i][j][0] = 0
                result[i][j][1] = 0
    return result.tolist()

#mean_and_variance = Get_mean_and_variance()

def Get_likelihood(mean_and_variance, col, label, xi):
    # data 是输入的数据x(二维矩阵)，y是label(一维list), i是特征的序号， xi是具体值， label是label的值（0,1）
    return np.exp(-0.5 * ((xi - mean_and_variance[label][col][0]) / mean_and_variance[label][col][1])) / (np.sqrt(2 * np.pi * mean_and_variance[label][col][1]))


if __name__ == '__main__':
    url = "adult.test"
    X,y = offer_data(url)


    print(y[:20])
    print(X[:20])
