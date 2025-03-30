import numpy as np

class DataLoader:
    def __init__(self, train_data_path, train_label_path,
                test_data_path, test_label_path, num_classes=10):
        self.num_classes = num_classes
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

        # 对训练数据进行归一化（标准化），用训练集的均值和标准差
        self.mean = np.mean(self.train_data, axis=0, keepdims=True)
        self.std = np.std(self.train_data, axis=0, keepdims=True)
        # 防止除0
        self.std[self.std == 0] = 1e-8
        self.train_data = (self.train_data - self.mean) / self.std
        self.test_data = (self.test_data - self.mean) / self.std

        # 将标签转换为 one-hot 编码
        self.train_label = self.to_one_hot(self.train_label, self.num_classes)
        self.test_label = self.to_one_hot(self.test_label, self.num_classes)

    def to_one_hot(self, labels, num_classes):
        """
        将标签转换为 one-hot 编码
        :param labels: 整数标签数组，形状 (N, 1) 或 (N,)
        :param num_classes: 类别数
        :return: one-hot 编码数组，形状 (N, num_classes)
        """
        labels = labels.flatten().astype(np.int32)
        one_hot = np.eye(num_classes)[labels]
        return one_hot

    def get_train_data(self):
        return self.train_data

    def get_train_labels(self):
        return self.train_label

    def get_test_data(self):
        return self.test_data

    def get_test_labels(self):
        return self.test_label

    def iter_train(self, shuffle=False):
        return _DataIterable(self.train_data, self.train_label, shuffle)

    def iter_test(self, shuffle=False):
        return _DataIterable(self.test_data, self.test_label, shuffle)

    def batch_generator(self, mode='train', batch_size=32, shuffle=False):
        """
        生成 mini-batch 的迭代器
        参数:
            mode: 'train' 或 'test'
            batch_size: 每个批次样本数
            shuffle: 是否打乱数据（训练时一般为 True）
        返回:
            每次 yield 一个 (data_batch, label_batch)
        """
        if mode == 'train':
            data = self.train_data
            labels = self.train_label
        elif mode == 'test':
            data = self.test_data
            labels = self.test_label
        else:
            raise ValueError("mode must be 'train' or 'test'")

        num = data.shape[0]
        indices = np.arange(num)
        if shuffle:
            np.random.shuffle(indices)

        for start_idx in range(0, num, batch_size):
            end_idx = min(start_idx + batch_size, num)
            batch_indices = indices[start_idx: end_idx]
            yield data[batch_indices], labels[batch_indices]

class _DataIterable:
    def __init__(self, data, label, shuffle=False):
        self.data = data
        self.label = label
        self.shuffle = shuffle
        self.indices = np.arange(len(data))
        if self.shuffle:
            np.random.shuffle(self.indices)
        self.index = 0

    def reset(self):
        """
        重置迭代器：将索引归零，并根据需要重新打乱顺序。
        实现迭代器复用。
        """
        self.index = 0
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __iter__(self):
        self.index = 0
        if self.shuffle:
            np.random.shuffle(self.indices)
        return self

    def __next__(self):
        if self.index >= len(self.data):
            raise StopIteration
        current_index = self.indices[self.index]
        data = self.data[current_index]
        label = self.label[current_index]
        self.index += 1
        return data, label

    def __len__(self):
        return len(self.label)