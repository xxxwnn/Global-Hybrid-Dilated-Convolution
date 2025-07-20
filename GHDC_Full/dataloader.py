import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

# dt = pd.read_csv(r'.\data\cirrhosis.csv', header=None)
# dt = pd.read_excel(r'.\data2\cirrhosis.xlsx', header=None)

datasets = "wt2d"
model_path = fr'.\normalized_clr\RF\{datasets}\pth\best_model_{datasets}_crucial.pth'
# model_path = fr'C:\Users\xxwn\Desktop\bio\Gut_flora_v1\data.5\{datasets}\best_model_{datasets}_crucial.pth'
dt = pd.read_excel(fr'C:\Users\xxwn\Desktop\bio\Gut_flora_v1\data2\{datasets}.xlsx', header=None)
# dt = pd.read_excel(fr'.\data.5\{datasets}.xlsx', header=None)

df1 = dt.iloc[:, :]

# Extract numeric labels (0 and 1)
# numeric_labels = df1.iloc[0, :].apply(lambda x: 0 if x[0] == 'n' else 1).tolist()
numeric_labels = df1.iloc[0, 1:].apply(lambda x: 0 if 'n' in str(x).lower() else 1).tolist()
# numeric_labels = dt.iloc[0, 1:].apply(lambda x: 1 if str(x).lower() == 'cancer' else 0).tolist()
# print(numeric_labels)
# Drop the first row to keep only the features
df1 = df1.iloc[1:, 1:]
# print(df1)
# Convert features to NumPy array and transpose
x = df1.values.astype(np.float32).T
# indices = np.random.permutation(x.shape[1])  # x.shape[1] 是 x 的列数


# x = x[:, indices]
# z = df1.values.astype(np.float32)
# n_features = z.shape[0]
# indices = np.arange(n_features).astype(np.float32)  # Create an index for each feature
# indices = [x / 10000 for x in range(0, n_features)]
# Expand dimensions of x and indices to enable concatenation
# Reshape x to (number_of_samples, n_features, 1)
# x_expanded = z[np.newaxis, ...]  # (1, number_of_features, number_of_samples)
# print(x_expanded.shape)
# Reshape indices to (number_of_samples, n_features, 1)
# indices_expanded = np.expand_dims(indices, axis=1)  # (n_features, 1)
# indices_repeated = np.repeat(indices_expanded, x.shape[0], axis=1)  # (n_features, number_of_samples)
# indices_repeated = indices_repeated[np.newaxis, ...]  # (1, n_features, number_of_samples)

# print(indices_repeated.shape)
# Concatenate along the first axis to combine features and indices
# combined_data = np.concatenate((x_expanded, indices_repeated), axis=0)  # (2, n_features, number_of_samples)

# print(combined_data)
# Convert combined data to tensor
# combined_data_tensor = torch.tensor(combined_data, dtype=torch.float32)  # (2, n_features, number_of_samples)
# print(combined_data_tensor)
# combined_data_tensor.to_csv('combined.csv')
# df1 = pd.DataFrame(combined_data_tensor[0])
# df2 = pd.DataFrame(combined_data_tensor[1])
# df1.to_csv(r'combined.csv', index=False)
# df2.to_csv(r'combined2.csv', index=False)
# print(combined_data.shape)

# print(x)
# Convert numeric labels to a list of integers
y = [int(label) for label in numeric_labels]

# print(y)
# state = np.random.get_state()
train_X, test_X, train_y, test_y = train_test_split(x, y, test_size=0.2)
train_x, test_x, Train_y, Test_y = train_test_split(train_X, train_y, test_size=0.2)


# print(state)
# np.random.set_state(state)


# class MyDataset(Dataset):  # 继承Dataset
#     def __init__(self, data, label):
#         self.x = torch.tensor(data)
#         self.y = torch.tensor(label)
#
#     def __len__(self):  # 返回整个数据集的大小
#         return len(self.x)
#
#     def __getitem__(self, index):  # 根据索引index返回图像及标签
#         return self.x[index], self.y[index]

class MyDataset(Dataset):  # MyDataset+CLR
    def __init__(self, data, label):
        # CLR 预处理 - 修正版本
        data = np.array(data, dtype=np.float32)

        # 确保数据为正数，避免log(0)
        data = data + 1e-8

        # 计算每个样本的几何均值
        # 对于每个样本（行），计算所有OTU的几何均值
        geometric_mean = np.exp(np.mean(np.log(data), axis=1, keepdims=True))

        # CLR变换：log(x_i / geometric_mean)
        clr_data = np.log(data / geometric_mean)

        self.x = torch.tensor(clr_data, dtype=torch.float32)
        self.y = torch.tensor(label, dtype=torch.long)

    def __len__(self):  # 返回整个数据集的大小
        return len(self.x)

    def __getitem__(self, index):  # 根据索引index返回图像及标签
        return self.x[index], self.y[index]
