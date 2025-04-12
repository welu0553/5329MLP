import numpy as np


class DataLoader:
    def __init__(self, train_data_path, train_label_path, test_data_path, test_label_path, num_classes=10):
        """
        Load training and testing datasets from given .npy files and preprocess them.

        Args:
            train_data_path (str): Path to training data (.npy)
            train_label_path (str): Path to training labels (.npy)
            test_data_path (str): Path to test data (.npy)
            test_label_path (str): Path to test labels (.npy)
            num_classes (int): Number of classification classes
        """
        self.num_classes = num_classes
        self.__data_read(train_data_path, train_label_path, test_data_path, test_label_path)

    def __str__(self):
        """
        Format and return dataset information as a summary table including shape, dtype, mean, std, min, and max.
        """
        def stats(arr):
            return float(np.mean(arr)), float(np.std(arr)), float(np.min(arr)), float(np.max(arr))

        info = [
            ('Train', 'Data', self.train_data, self.train_data.dtype),
            ('Train', 'Label', self.train_label_origin, self.train_label.dtype),
            ('Test', 'Data', self.test_data, self.test_data.dtype),
            ('Test', 'Label', self.test_label_origin, self.test_label.dtype)
        ]

        table_rows = []
        for set_name, type_name, arr, dtype in info:
            shape = arr.shape
            mean, std, min_val, max_val = stats(arr)
            table_rows.append((set_name, type_name, shape, str(dtype), mean, std, min_val, max_val))

        headers = ['Set', 'Type', 'Shape', 'Dtype', 'Mean', 'Std', 'Min', 'Max']

        col1_width = max(len(row[0]) for row in table_rows + [("Set", "", "", "", 0, 0, 0, 0)])
        col2_width = max(len(row[1]) for row in table_rows + [("", "Type", "", "", 0, 0, 0, 0)])
        col3_width = max(len(str(row[2])) for row in table_rows + [("", "", "Shape", "", 0, 0, 0, 0)])
        col4_width = max(len(row[3]) for row in table_rows + [("", "", "", "Dtype", 0, 0, 0, 0)])
        col5_width = col6_width = col7_width = col8_width = 9

        header_fmt = (
            f"| {{:<{col1_width}}} | {{:<{col2_width}}} | {{:<{col3_width}}} | "
            f"{{:<{col4_width}}} | {{:>{col5_width}}} | {{:>{col6_width}}} | "
            f"{{:>{col7_width}}} | {{:>{col8_width}}} |"
        )

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

        lines = [separator, header_fmt.format(*headers), separator]
        last_set = None
        for row in table_rows:
            display_set = row[0] if row[0] != last_set else ''
            lines.append(
                f"| {display_set:<{col1_width}} | {row[1]:<{col2_width}} | {str(row[2]):<{col3_width}} | "
                f"{row[3]:<{col4_width}} | {row[4]:>{col5_width}.4f} | {row[5]:>{col6_width}.4f} | "
                f"{row[6]:>{col7_width}.4f} | {row[7]:>{col8_width}.4f} |"
            )
            last_set = row[0]
        lines.append(separator)
        return "\n" + "\n".join(lines)

    def __data_read(self, train_data_path, train_label_path, test_data_path, test_label_path):
        """
        Load and normalize datasets, convert labels to one-hot encoding.
        """
        self.train_data = np.load(train_data_path)
        self.train_label = np.load(train_label_path)
        self.train_label_origin = self.train_label
        self.test_data = np.load(test_data_path)
        self.test_label = np.load(test_label_path)
        self.test_label_origin = self.test_label

        self.mean = np.mean(self.train_data, axis=0, keepdims=True)
        self.std = np.std(self.train_data, axis=0, keepdims=True)
        self.std[self.std == 0] = 1e-8  # Avoid division by zero

        self.train_data = (self.train_data - self.mean) / self.std
        self.test_data = (self.test_data - self.mean) / self.std

        self.train_label = self.to_one_hot(self.train_label, self.num_classes)
        self.test_label = self.to_one_hot(self.test_label, self.num_classes)

    def to_one_hot(self, labels, num_classes):
        """
        Convert integer labels to one-hot encoded matrix.

        Args:
            labels (ndarray): Label vector (N,)
            num_classes (int): Total number of classes

        Returns:
            ndarray: One-hot encoded labels (N, num_classes)
        """
        labels = labels.flatten().astype(np.int32)
        one_hot = np.eye(num_classes)[labels]
        return one_hot

    def get_train_data(self):
        """Return normalized training data"""
        return self.train_data

    def get_train_labels(self):
        """Return one-hot encoded training labels"""
        return self.train_label

    def get_test_data(self):
        """Return normalized test data"""
        return self.test_data

    def get_test_labels(self):
        """Return one-hot encoded test labels"""
        return self.test_label

    def iter_train(self, shuffle=False):
        """Return a sample-level iterator over training data"""
        return _DataIterable(self.train_data, self.train_label, shuffle)

    def iter_test(self, shuffle=False):
        """Return a sample-level iterator over test data"""
        return _DataIterable(self.test_data, self.test_label, shuffle)

    def batch_generator(self, mode='train', batch_size=32, shuffle=False):
        """
        Yield mini-batches of data.

        Args:
            mode (str): 'train' or 'test'
            batch_size (int): Number of samples per batch
            shuffle (bool): Whether to shuffle the data

        Yields:
            Tuple of (data_batch, label_batch)
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
        """
        A sample-level iterator used for evaluation or debugging.

        Args:
            data (ndarray): Data samples
            label (ndarray): Corresponding labels
            shuffle (bool): Whether to shuffle on reset
        """
        self.data = data
        self.label = label
        self.shuffle = shuffle
        self.indices = np.arange(len(data))
        if self.shuffle:
            np.random.shuffle(self.indices)
        self.index = 0

    def reset(self):
        """
        Reset the iterator to the beginning and reshuffle if needed.
        """
        self.index = 0
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __iter__(self):
        """
        Return iterator instance
        """
        self.index = 0
        if self.shuffle:
            np.random.shuffle(self.indices)
        return self

    def __next__(self):
        """
        Return the next sample (data, label).
        """
        if self.index >= len(self.data):
            raise StopIteration
        current_index = self.indices[self.index]
        data = self.data[current_index]
        label = self.label[current_index]
        self.index += 1
        return data, label

    def __len__(self):
        """Return the number of samples"""
        return len(self.label)
