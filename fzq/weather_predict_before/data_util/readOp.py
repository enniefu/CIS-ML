import re

def readOpcal(url):
    with open(url,'r') as f:
        data = []
        # lines = f.readline() #读第一行，删掉表头
        while True:
            lines = f.readline()  # 整行读取数据
            if not lines:
                break

            #处理一行的数据
            dataline=[]
            texts = lines.split(' ')

            #匹配本行过滤字段中无意义的符号
            for text in texts:
                if text==' ' or text=='':
                    continue
                else:
                    result = re.match(r"[0-9]*.?[0-9]*", text)
                    try:
                        dataline.append(float(result.group()))
                    except:
                        print(result)

            #删除无意义的字段
            process_dataline=[]
            for i in range(len(dataline)):
                if i not in (0,1,2,4,6,8,10,12,14):
                    process_dataline.append(dataline[i])
            data.append(process_dataline)

        return data

if __name__ == '__main__':
    url=r"D:\010010 1987-2018.txt"
    texts=readOpcal(url)
    for i in texts:
        print(i)
    print(len(texts))