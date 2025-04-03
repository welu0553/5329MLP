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
        # 辅助函数：计算数组统计量
        def stats(arr):
            return float(np.mean(arr)), float(np.std(arr)), float(np.min(arr)), float(np.max(arr))

        # 数据准备：每行包括 set 名称、类型、原数组、dtype
        info = [
            ('Train', 'Data', self.train_data, self.train_data.dtype),
            ('Train', 'Label', self.train_label_origin, self.train_label.dtype),
            ('Test', 'Data', self.test_data, self.test_data.dtype),
            ('Test', 'Label', self.test_label_origin, self.test_label.dtype)
        ]

        # 构造每行信息，并计算统计量
        table_rows = []
        for set_name, type_name, arr, dtype in info:
            shape = arr.shape
            mean, std, min_val, max_val = stats(arr)
            table_rows.append((set_name, type_name, shape, str(dtype), mean, std, min_val, max_val))

        # 表头定义
        headers = ['Set', 'Type', 'Shape', 'Dtype', 'Mean', 'Std', 'Min', 'Max']

        # 计算前4列（文本型）的最大宽度（考虑表头）
        col1_width = max(len(row[0]) for row in table_rows + [("Set", "", "", "", 0, 0, 0, 0)])
        col2_width = max(len(row[1]) for row in table_rows + [("", "Type", "", "", 0, 0, 0, 0)])
        col3_width = max(len(str(row[2])) for row in table_rows + [("", "", "Shape", "", 0, 0, 0, 0)])
        col4_width = max(len(row[3]) for row in table_rows + [("", "", "", "Dtype", 0, 0, 0, 0)])

        # 对于数字列，设置固定宽度（例如 9 个字符）
        col5_width = 9  # Mean
        col6_width = 9  # Std
        col7_width = 9  # Min
        col8_width = 9  # Max

        # 构造格式化字符串：头部和数据行使用相同格式
        header_fmt = (
            f"| {{:<{col1_width}}} | {{:<{col2_width}}} | {{:<{col3_width}}} | "
            f"{{:<{col4_width}}} | {{:>{col5_width}}} | {{:>{col6_width}}} | "
            f"{{:>{col7_width}}} | {{:>{col8_width}}} |"
        )

        # 构造分隔行
        separator = (
                '+' + '-' * (col1_width + 2) +
                '+' + '-' * (col2_width + 2) +
                '+' + '-' * (col3_width + 2) +
                '+' + '-' * (col4_width + 2) +
                '+' + '-' * (col5_width + 2) +
                '+' + '-' * (col6_width + 2) +
                '+' + '-' * (col7_width + 2) +
                '+' + '-' * (col8_width + 2) + '+'
        )

        # 构建输出表格
        lines = [separator]
        lines.append(header_fmt.format(*headers))
        lines.append(separator)

        last_set = None
        for row in table_rows:
            # 若连续的两行所属的 set 名称相同，第一列只显示一次
            display_set = row[0] if row[0] != last_set else ''
            lines.append(
                f"| {display_set:<{col1_width}} | {row[1]:<{col2_width}} | {str(row[2]):<{col3_width}} | "
                f"{row[3]:<{col4_width}} | {row[4]:>{col5_width}.4f} | {row[5]:>{col6_width}.4f} | "
                f"{row[6]:>{col7_width}.4f} | {row[7]:>{col8_width}.4f} |"
            )
            last_set = row[0]

        lines.append(separator)
        return "\n" + "\n".join(lines)

    # def __str__(self):
    #     l1 = f'Train data  : shape: {self.train_data.shape}, dtype: {self.train_data.dtype}'
    #     l2 = f'      labels:        {self.train_label.shape},          {self.train_label.dtype}'
    #     l3 = f'Test  data  : shape: {self.test_data.shape}, dtype: {self.test_data.dtype}'
    #     l4 = f'      labels:        {self.test_label.shape},          {self.test_label.dtype}'
    #     sep = (len(max(l1, l2, l3, l4))) * '-'
    #     total_len = len(sep)
    #     l1, l2, l3, l4 = ['|' + x + (total_len - len(x)) * ' ' + '|' + '\n' for x in [l1, l2, l3, l4]]
    #     separator = '|' + sep + '|\n'
    #     s = '\n' + separator + l1 + l2 + l3 + l4 + separator
    #     return s

    def __data_read(self, train_data_path, train_label_path, test_data_path, test_label_path):
        self.train_data = np.load(train_data_path)
        self.train_label = np.load(train_label_path)
        self.train_label_origin = self.train_label
        self.test_data = np.load(test_data_path)
        self.test_label = np.load(test_label_path)
        self.test_label_origin = self.test_label

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