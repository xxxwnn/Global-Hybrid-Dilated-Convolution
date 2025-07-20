import time
import matplotlib.pyplot as plt
import numpy as np
import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from optuna.samplers import TPESampler
from sklearn.metrics import f1_score, roc_auc_score, recall_score, precision_score, precision_recall_curve, \
    auc as pr_auc
from sklearn.metrics import matthews_corrcoef
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import os
import csv

from GHDC import CNNModel
from dataloader import MyDataset, train_X, test_X, train_y, test_y, datasets, model_path
from objectives import objective

start_time = time.time()
device = torch.device("cuda")
print(torch.cuda.is_available())
# device = torch.device('cuda:1') #数字切换卡号
print(device)


# def train(epoch):
#     global total_train_step
#     total_train_step = 0
#     zero_value_counts = []
#     for data in train_loader:
#         # print(data)
#         imgs, targets = data
#         imgs = imgs.unsqueeze(1).to(device)
#         perm = torch.randperm(imgs.size(0))
#         imgs = imgs[perm]
#         targets = targets.to(torch.long).to(device)
#         targets = targets[perm]
#         optimizer.zero_grad()
#         # outputs = net(imgs, targets)
#         outputs = net(imgs)
#         loss = loss_fn(outputs, targets)
#         loss.backward()
#         optimizer.step()
#         if total_train_step % 200 == 0:
#             print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
#                 epoch, total_train_step, len(dtrain),
#                 100. * total_train_step / len(dtrain), loss.item()))
#         writer.add_scalar('loss', loss.item(), total_train_step)
#         total_train_step += 1


def train(epoch):
    global total_train_step
    total_train_step = 0
    total_loss = 0  # 用于计算每个epoch的平均loss
    batch_count = 0
    correct = 0
    total = 0

    for data in train_loader:
        imgs, targets = data
        # print(imgs.shape)
        imgs = imgs.unsqueeze(1).to(device)
        # print(imgs.shape)
        targets = targets.to(torch.long).to(device)
        optimizer.zero_grad()
        # print(imgs.shape, targets.shape)
        outputs = net(imgs)
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        batch_count += 1
        total += targets.size(0)
        _, predicted = torch.max(outputs.data, 1)
        correct += (predicted == targets).sum().item()

        if total_train_step % 200 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, total_train_step, len(dtrain),
                100. * total_train_step / len(dtrain), loss.item()))

        writer.add_scalar('loss', loss.item(), total_train_step)
        total_train_step += 1
    print(correct / total)

    # 返回当前epoch的平均loss
    return total_loss / batch_count


# Test---------------------------------

def eval():
    correct = 0
    total = 0
    all_labels = []
    all_predictions = []
    all_probabilities = []

    with torch.no_grad():
        for data in test_loader:
            imgs, targets = data
            # print(imgs.shape)
            imgs = imgs.unsqueeze(1).to(device)  # 增加一个通道维度
            targets = targets.to(torch.long).to(device)
            # outputs = net(imgs, targets)
            outputs = net(imgs)
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
            print('Result:{}, True:{}'.format(predicted.tolist(), targets.tolist()))

    # 计算 ROC AUC
    rocauc1 = roc_auc_score(all_labels, all_probabilities)
    precision_curve, recall_curve, _ = precision_recall_curve(all_labels, all_probabilities)
    aupr = pr_auc(recall_curve, precision_curve)
    print('Test Accuracy: {}/{} ({:.0f}%)'.format(correct, total, 100. * correct / total))
    mcc = matthews_corrcoef(all_labels, all_predictions)
    recall = recall_score(all_labels, all_predictions, average='binary')
    precision = precision_score(all_labels, all_predictions, average='binary')
    # 计算 F1 分数
    macro_f1 = f1_score(all_labels, all_predictions, average='binary')
    return correct / total, rocauc1, recall, precision, mcc, aupr, macro_f1


if __name__ == '__main__':
    # 结果保存文件名
    result_file = './data.5/metrics_summary_runs.csv'
    dataset_name = datasets
    n_runs = 10
    all_results = []
    for run_id in range(1, n_runs + 1):
        # 每次都重新划分数据
        from dataloader import train_X, test_X, train_y, test_y, MyDataset
        dtrain = MyDataset(train_X, train_y)
        dtest = MyDataset(test_X, test_y)
        train_loader = DataLoader(dataset=dtrain, batch_size=16, shuffle=True, num_workers=0, drop_last=True)
        test_loader = DataLoader(dataset=dtest, batch_size=16, shuffle=False, num_workers=0, drop_last=False)
        # 初始化模型和优化器
        net = CNNModel().to(device)
        loss_fn = nn.CrossEntropyLoss().to(device)
        learning_rate = 0.0001  # 可根据需要调整
        epoch = 100  # 可根据需要调整
        optimizer = optim.Adam(net.parameters(), lr=learning_rate)
        writer = SummaryWriter(log_dir=f'logs/{time.strftime("%Y%m%d-%H%M%S")}_run{run_id}')
        best_score = None
        counter = 0
        patience = 40
        test_accuracies = []
        AUC = []
        F1_scores = []
        MCC_scores = []
        Recall_scores = []
        Precision_scores = []
        AUPR_scores = []
        for i in range(1, epoch + 1):
            train_loader = DataLoader(dataset=dtrain, batch_size=16, shuffle=True, num_workers=0, drop_last=True)
            train(i)
            test_accuracy, auc, recall, precision, mcc, aupr, f1 = eval()
            test_accuracies.append(test_accuracy)
            AUC.append(auc)
            MCC_scores.append(mcc)
            Recall_scores.append(recall)
            Precision_scores.append(precision)
            F1_scores.append(f1)
            AUPR_scores.append(aupr)
            writer.add_scalar('test_accuracy', test_accuracy, total_train_step)
            if best_score is None or test_accuracy >= best_score:
                best_score = test_accuracy
                counter = 0
                # torch.save(net.state_dict(), model_path)  # 可选：保存最优模型
            else:
                counter += 1
            if counter >= patience:
                break
        # 记录本次run的最优结果
        result = {
            'dataset': dataset_name,
            'run_id': run_id,
            'epoch': epoch,
            'learning_rate': learning_rate,
            'Best Accuracy': max(test_accuracies),
            'Best AUC': max(AUC),
            'Avg Precision': np.mean(Precision_scores),
            'Avg Recall': np.mean(Recall_scores),
            'Avg F1': np.mean(F1_scores),
            'Avg MCC': np.mean(MCC_scores),
            'Avg AUPR': np.mean(AUPR_scores)
        }
        all_results.append(result)
        # 追加写入CSV
        file_exists = os.path.isfile(result_file)
        with open(result_file, mode='a', newline='') as file:
            writer_csv = csv.writer(file)
            if not file_exists:
                writer_csv.writerow(list(result.keys()))
            writer_csv.writerow(list(result.values()))
    print(f"全部完成，结果已保存到 {result_file}")
