from e_util import offer_data
import torch
from torch.utils.data import  DataLoader
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, url, train=True):
        # TODO
        # 1. Initialize file paths or a list of file names.
        self.train = train
        X,y = offer_data(url,7,0,1)

        if self.train:
            self.X_train = torch.from_numpy(X[:8000])
            self.y_train = torch.from_numpy(y[:8000])
            self.len = self.X_train.shape[0]
        else:
            self.X_test = torch.from_numpy(X[8000:10000])
            self.y_test = torch.from_numpy(y[8000:10000])
            self.len = self.X_test.shape[0]



    def __getitem__(self, index):
        # TODO
        # 1. Read one data from file (e.g. using numpy.fromfile, PIL.Image.open).
        # 2. Preprocess the data (e.g. torchvision.Transform).
        # 3. Return a data pair (e.g. image and label).
        if self.train:
            return self.X_train[index], self.y_train[index]
        else:
            return self.X_test[index],self.y_test[index]
        pass
    def __len__(self):
        return self.len

if __name__ == '__main__':
    dealDataset = CustomDataset(url=r"D:\010010 1987-2018.txt")
    train_loader2 = DataLoader(dataset=dealDataset,
                               batch_size=32,
                               shuffle=True)
    for epoch in range(2):
        for i, data in enumerate(train_loader2):
            # 将数据从 train_loader 中读出来,一次读取的样本数是32个
            inputs, labels = data

            # 接下来就是跑模型的环节了，我们这里使用print来代替
            print("epoch：", epoch, "的第", i, "个inputs", inputs.data.size(), "labels", labels.data.size())
