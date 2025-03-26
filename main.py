"""
流程:
    1. 数据集导入模块

    2. MLP模型模块

    3. 损失函数优化器模块

    4. 训练 / 测试模块

"""
import data_loader

def main():
    DataLoader = data_loader.DataLoader(
        train_data_path='./Assignment1-Dataset/train_data.npy',
        train_label_path='./Assignment1-Dataset/train_label.npy',
        test_data_path='./Assignment1-Dataset/test_data.npy',
        test_label_path='./Assignment1-Dataset/test_label.npy'
    )
    print(DataLoader)

    pass

if __name__ == '__main__':
    main()