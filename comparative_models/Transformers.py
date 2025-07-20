import numpy as np
import optuna
import pandas as pd
from sklearn.model_selection import train_test_split
from torch import optim
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc as pr_auc, matthews_corrcoef, recall_score, \
    precision_score, f1_score

from objectives import AdaptiveFocalLoss
import os

# 数据读取
dataset = 'wt2d'
# dt = pd.read_excel(r'.\data1\crucial germs cirrhosis.xlsx', header=None)
dt = pd.read_excel(fr'C:\Users\xxwn\Desktop\bio\Gut_flora_v1\data2\{dataset}.xlsx', header=None)
df1 = dt.iloc[:, :]

# 提取标签并进行编码
numeric_labels = df1.iloc[0, 1:].apply(lambda x: 0 if 'n' in str(x).lower() else 1).tolist()
# print(numeric_labels)
df1 = df1.iloc[1:, 1:]  # 删除第一行和第一列，保留数值特征
x = df1.values.astype(np.float32).T  # 转置为 [样本数, 特征数]
y = [int(label) for label in numeric_labels]  # 转换标签为整数列表


def clr_transform(X, eps=1e-8):
    X = np.array(X)
    X = np.where(X <= 0, eps, X)  # 防止log(0)
    log_X = np.log(X)
    mean_log = np.mean(log_X, axis=1, keepdims=True)
    clr_X = log_X - mean_log
    return clr_X


x = clr_transform(x)
# 模型超参数

sequence_length = x.shape[1]  # 特征数即序列长度
batch_size = 32
device = torch.device("cuda" if torch.cuda.is_available() else "")


# 自定义数据集
class MyDataset(Dataset):
    def __init__(self, data, label):
        self.x = torch.tensor(data, dtype=torch.float32)  # 转换为张量
        self.y = torch.tensor(label, dtype=torch.long)  # 标签为整数

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return self.x[index], self.y[index]


# 数据加载
dataset = MyDataset(x, y)
# dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
input_dim = sequence_length  # 输入特征数
num_classes = 2  # 假设二分类
train_X, test_X, train_y, test_y = train_test_split(x, y, test_size=0.2)
dtrain = MyDataset(train_X, train_y)
dtest = MyDataset(test_X, test_y)
train_loader = DataLoader(dataset=dtrain, batch_size=16, shuffle=True, num_workers=0, drop_last=True)
test_loader = DataLoader(dataset=dtest, batch_size=16, shuffle=False, num_workers=0, drop_last=False)


def eval(net, test_loader, device):
    correct = 0
    total = 0
    all_labels = []
    all_predictions = []
    all_probabilities = []

    with torch.no_grad():
        for data in test_loader:
            imgs, targets = data
            imgs = imgs.to(device)  # 确保输入数据在正确的设备上
            targets = targets.to(torch.long).to(device)
            outputs = net(imgs)  # 假设你的模型不需要targets作为输入
            _, predicted = torch.max(outputs.data, 1)

            # 保存每个 batch 的标签和预测值
            all_labels.extend(targets.cpu().tolist())
            all_predictions.extend(predicted.cpu().tolist())

            # 保存概率用于计算 roc_auc_score
            probabilities = torch.softmax(outputs, dim=1)
            if probabilities.shape[1] == 2:
                probabilities = probabilities[:, 1]  # 获取正类的概率
            all_probabilities.extend(probabilities.cpu().tolist())

            total += targets.size(0)
            correct += (predicted == targets).sum().item()

    # 计算 ROC AUC
    rocauc1 = roc_auc_score(all_labels, all_probabilities)
    precision_curve, recall_curve, _ = precision_recall_curve(all_labels, all_probabilities)
    aupr = pr_auc(recall_curve, precision_curve)
    print('Test Accuracy: {}/{} ({:.0f}%)'.format(correct, total, 100. * correct / total))
    mcc = matthews_corrcoef(all_labels, all_predictions)
    recall = recall_score(all_labels, all_predictions, average='binary')
    precision = precision_score(all_labels, all_predictions, average='binary', zero_division=1)
    macro_f1 = f1_score(all_labels, all_predictions, average='binary')

    return correct / total, rocauc1, recall, precision, mcc, aupr, macro_f1


# 定义模型
class TransformerModel(nn.Module):
    def __init__(self, embedding_dim, num_classes, nhead, num_layers):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Linear(1, embedding_dim)  # 将输入特征映射到嵌入维度
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=nhead)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(embedding_dim, num_classes)  # 全连接层，分类输出

    def forward(self, x):
        # 输入形状: [batch_size, sequence_length]
        # print(x.shape)
        x = x.unsqueeze(-1)
        x = self.embedding(x)  # [batch_size, sequence_length, embedding_dim]
        # print(x.shape)
        x = x.permute(1, 0, 2)  # 转换为 [sequence_length, batch_size, embedding_dim] 以适配 Transformer
        # print(x.shape)
        x = self.transformer_encoder(x)  # 通过 Transformer 编码器
        x = x.mean(dim=0)  # 对序列长度维度取平均，形状为 [batch_size, embedding_dim]
        x = self.fc(x)  # 全连接层输出
        return x


if __name__ == '__main__':
    # 结果保存文件名
    result_file = r'C:\Users\xxwn\Desktop\bio\Gut_flora_v1\Transformers\results.csv'
    all_results = []
    for run_id in range(1, 11):
        # 每次都重新划分数据
        train_X, test_X, train_y, test_y = train_test_split(x, y, test_size=0.2)
        dtrain = MyDataset(train_X, train_y)
        dtest = MyDataset(test_X, test_y)
        train_loader = DataLoader(dataset=dtrain, batch_size=16, shuffle=True, num_workers=0, drop_last=True)
        test_loader = DataLoader(dataset=dtest, batch_size=16, shuffle=False, num_workers=0, drop_last=False)


        # Optuna调参
        def objective(trial):
            def eval_model(model, data_loader, criterion, device):
                model.eval()
                total_loss = 0
                correct = 0
                total = 0
                all_labels = []
                all_predictions = []
                all_probabilities = []
                auc_scores = []
                with torch.no_grad():
                    for data, targets in data_loader:
                        data, targets = data.to(device), targets.to(device)
                        outputs = model(data)
                        loss = criterion(outputs, targets)
                        total_loss += loss.item() * targets.size(0)
                        _, predicted = torch.max(outputs.data, 1)
                        correct += (predicted == targets).sum().item()
                        total += targets.size(0)
                        probabilities = torch.softmax(outputs, dim=1)[:, 1]
                        all_labels.extend(targets.cpu().tolist())
                        all_probabilities.extend(probabilities.cpu().tolist())
                        all_predictions.extend(predicted.cpu().tolist())
                avg_loss = total_loss / total if total > 0 else 0
                accuracy = correct / total if total > 0 else 0
                auc = roc_auc_score(all_labels, all_probabilities)
                auc_scores.append(auc)
                if len(auc_scores) > 10:
                    auc_scores.pop(0)
                avg_auc = sum(auc_scores) / len(auc_scores)
                return avg_auc

            embedding_dim = trial.suggest_int('embedding_dim', 64, 512)
            nhead = trial.suggest_int('nhead', 2, 6)
            num_layers = trial.suggest_int('num_layers', 1, 4)
            learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
            Epoches = trial.suggest_int('Epoches', 10, 100)
            if embedding_dim % nhead != 0:
                embedding_dim = (embedding_dim // nhead) * nhead
            model = TransformerModel(embedding_dim, num_classes, nhead, num_layers).to(device)
            criterion = AdaptiveFocalLoss()
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
            model.train()
            for epoch in range(Epoches):
                for batch_idx, (data, target) in enumerate(train_loader):
                    data = data.to(device)
                    target = target.to(device)
                    optimizer.zero_grad()
                    output = model(data)
                    loss = criterion(output, target)
                    loss.backward()
                    optimizer.step()
            model.eval()
            auc = eval_model(model, test_loader, criterion, device)
            return auc


        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=10)
        best_params = study.best_params
        embedding_dims = best_params["embedding_dim"]
        nhead = best_params["nhead"]
        if embedding_dims % nhead != 0:
            embedding_dims = (embedding_dims // nhead) * nhead
        num_layers = best_params["num_layers"]
        learning_rate = best_params["learning_rate"]
        epoches = best_params['Epoches']
        model = TransformerModel(embedding_dims, num_classes, nhead, num_layers).to(device)
        criterion = AdaptiveFocalLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        for epoch in range(epoches):
            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                outputs = model(batch_x).to(device)
                loss = criterion(outputs, batch_y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        accuracy, rocauc, recall, precision, mcc, aupr, f1 = eval(model, test_loader, device)
        # 记录结果
        result = {
            'dataset': dataset,
            'run_id': run_id,
            'embedding_dim': embedding_dims,
            'nhead': nhead,
            'num_layers': num_layers,
            'learning_rate': learning_rate,
            'epoches': epoches,
            'Accuracy': accuracy,
            'AUC': rocauc,
            'Recall': recall,
            'Precision': precision,
            'F1': f1,
            'MCC': mcc,
            'AUPR': aupr
        }
        all_results.append(result)
        # 追加写入CSV
        df_result = pd.DataFrame([result])
        if not os.path.exists(result_file):
            df_result.to_csv(result_file, mode='w', index=False, encoding='utf-8-sig')
        else:
            df_result.to_csv(result_file, mode='a', index=False, header=False, encoding='utf-8-sig')
    print(f"全部完成，结果已保存到 {result_file}")
