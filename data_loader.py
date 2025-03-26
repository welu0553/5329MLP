import numpy as np

class DataLoader:
    def __init__(self, train_data_path, train_label_path,
                test_data_path, test_label_path):
        self.__data_read(
            train_data_path, train_label_path,
            test_data_path, test_label_path
        )

    def __str__(self):
        l1 = f'Train data  : shape: {self.train_data.shape}, dtype: {self.train_data.dtype}'
        l2 = f'      labels:        {self.train_label.shape},          {self.train_label.dtype}'
        l3 = f'Test  data  : shape: {self.test_data.shape}, dtype: {self.test_data.dtype}'
        l4 = f'      labels:        {self.test_label.shape},          {self.test_label.dtype}'
        sep = (len(max(l1, l2, l3, l4))) * '-'
        total_len = len(sep)
        l1, l2, l3, l4 = ['|' + x + (total_len - len(x)) * ' ' + '|' + '\n' for x in [l1, l2, l3, l4]]
        separator = '|' + sep + '|\n'
        s = '\n' + separator + l1 + l2 + l3 + l4 + separator
        return s

    def __data_read(self, train_data_path, train_label_path, test_data_path, test_label_path):
        self.train_data = np.load(train_data_path)
        self.train_label = np.load(train_label_path)
        self.test_data = np.load(test_data_path)
        self.test_label = np.load(test_label_path)

    def get_train(self, t):
        if t == 'data':
            return self.train_data
        elif t == 'label':
            return self.train_label
        else:
            raise TypeError("Wrong parameter, only allowed 'data' or 'label'")

    def get_test(self, t):
        if t == 'data':
            return self.test_data
        elif t == 'label':
            return self.test_label
        else:
            raise TypeError("Wrong parameter, only allowed 'data' or 'label'")

    def iter_train(self):
        i = DataIterable(self.train_data, self.train_label)
        return i

    def iter_test(self):
        i = DataIterable(self.test_data, self.test_label)
        return i

class DataIterable:
    def __init__(self, data, label):
        self.data = data
        self.label = label
        self.index = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.index >= len(self.data):
            raise StopIteration
        data = self.data[self.index]
        label = self.label[self.index]
        self.index += 1
        return data, label