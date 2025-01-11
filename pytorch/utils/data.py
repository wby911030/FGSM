from torch.utils.data import Dataset

class Data(Dataset):
    def __init__(self, data, labels):
        self.data = []
        if isinstance(data, list) or isinstance(data, tuple):
            for x in data:
                self.data.append(x)
        else:
            self.data.append(data)
        self.data.append(labels)

    def __len__(self):
        return len(self.data[-1])

    def __getitem__(self, idx):
        item = []
        for i in range(len(self.data)):
            item.append(self.data[i][idx])
        return item