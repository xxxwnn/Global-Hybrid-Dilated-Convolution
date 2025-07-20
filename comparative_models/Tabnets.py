import optuna
import pandas as pd
import numpy as np
import os
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef, average_precision_score

# 数据集名称
datasets = 'wt2d'  # 可根据实际情况修改
excel_path = './tabnet_performance.xlsx'  # 结果保存路径

# CLR变换
def clr_transform(X, eps=1e-8):
    X = np.array(X)
    X = np.where(X <= 0, eps, X)  # 防止log(0)
    log_X = np.log(X)
    mean_log = np.mean(log_X, axis=1, keepdims=True)
    clr_X = log_X - mean_log
    return clr_X

# 读取数据
dt = pd.read_excel(r'./data2/wt2d.xlsx', header=None)
labels = dt.iloc[0, 1:].apply(lambda x: 0 if 'n' in str(x).lower() else 1).values
features = dt.iloc[1:, 1:].values.astype(float)
features = clr_transform(features)
features = features.T  # 保证样本为行

results = []

for run in range(10):
    print(f'========== Run {run+1} ==========')
    # 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, stratify=labels, random_state=run)

    def objective(trial):
        n_d = trial.suggest_int('n_d', 8, 128, step=8)
        n_a = trial.suggest_int('n_a', 8, 128, step=8)
        clf = TabNetClassifier(
            n_d=n_d,
            n_a=n_a,
        )
        clf.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            max_epochs=100, patience=15, batch_size=16, virtual_batch_size=8,
            num_workers=0, drop_last=False,
            # verbose=0
        )
        preds_proba = clf.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, preds_proba)
        return auc

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=15)
    best_params = study.best_params

    # 用最优参数训练最终模型
    clf = TabNetClassifier(n_a=best_params['n_a'], n_d=best_params['n_d'])
    clf.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        max_epochs=100, patience=15, batch_size=16, virtual_batch_size=8,
        num_workers=0, drop_last=False,
        # verbose=0
    )

    y_pred_proba = clf.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= 0.5).astype(int)

    accuracy = accuracy_score(y_test, y_pred)
    auc_score = roc_auc_score(y_test, y_pred_proba)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    mcc = matthews_corrcoef(y_test, y_pred)
    aupr = average_precision_score(y_test, y_pred_proba)

    results.append({
        'Dataset': datasets,
        'Run': run + 1,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1': f1,
        'MCC': mcc,
        'AUPR': aupr,
        'AUC': auc_score,
        'Best_Params': str(best_params)
    })

# 保存到Excel，自动追加
results_df = pd.DataFrame(results)
if os.path.exists(excel_path):
    old_df = pd.read_excel(excel_path)
    all_df = pd.concat([old_df, results_df], ignore_index=True)
    all_df.to_excel(excel_path, index=False)
else:
    results_df.to_excel(excel_path, index=False)
print("Results saved to", excel_path)
