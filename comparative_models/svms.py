import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, matthews_corrcoef, precision_recall_curve, auc, \
    roc_auc_score, f1_score
from sklearn.utils import shuffle
import optuna
import os

# 数据集名称
datasets = 't2d'  # 可根据实际情况修改
excel_path = r'C:\Users\xxwn\Desktop\bio\Gut_flora_v1\SVM\svm_performance.xlsx'  # 结果保存路径


# 加入clr归一化
def clr_transform(X, eps=1e-8):
    X = np.array(X)
    X = np.where(X <= 0, eps, X)  # 防止log(0)
    log_X = np.log(X)
    mean_log = np.mean(log_X, axis=1, keepdims=True)
    clr_X = log_X - mean_log
    return clr_X


# 加载数据
file_path = fr'C:\Users\xxwn\Desktop\bio\Gut_flora_v1\data2\{datasets}.xlsx'
df = pd.read_excel(file_path, header=None)
X = df.iloc[1:, 1:]
y = df.iloc[0, 1:].apply(lambda x: 0 if 'n' in str(x).lower() else 1).tolist()
X = X.apply(pd.to_numeric, errors='coerce').fillna(0)
X = clr_transform(X.values)  # 加入clr归一化
X = X.T  # 保持样本为行，特征为列
y = np.array(y)

results = []

for run in range(10):
    print(f'========== Run {run + 1} ==========')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=run)


    def objective(trial):
        C = trial.suggest_loguniform("C", 0.001, 10)
        kernel = trial.suggest_categorical("kernel", ["linear", "rbf", "poly"])
        gamma = trial.suggest_loguniform("gamma", 0.0001, 1) if kernel != "linear" else "scale"

        svm_model = SVC(C=C, kernel=kernel, gamma=gamma, probability=True, random_state=42)
        svm_model.fit(X_train, y_train)
        y_pred = svm_model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        mcc = matthews_corrcoef(y_test, y_pred)
        y_prob = svm_model.predict_proba(X_test)[:, 1]
        auc1 = roc_auc_score(y_test, y_prob)
        precision, recall, _ = precision_recall_curve(y_test, y_prob)
        aupr = auc(recall, precision)
        return auc1


    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=15)
    best_params = study.best_params

    svm_best_model = SVC(
        C=best_params["C"],
        kernel=best_params["kernel"],
        gamma=best_params["gamma"] if best_params["kernel"] != "linear" else "scale",
        probability=True,
        random_state=42
    )
    svm_best_model.fit(X_train, y_train)
    y_pred_best = svm_best_model.predict(X_test)
    accuracy_best = accuracy_score(y_test, y_pred_best)
    mcc_best = matthews_corrcoef(y_test, y_pred_best)
    f1_best = f1_score(y_test, y_pred_best, zero_division=0)
    y_prob_best = svm_best_model.predict_proba(X_test)[:, 1]
    precision_best, recall_best, _ = precision_recall_curve(y_test, y_prob_best)
    aupr_best = auc(recall_best, precision_best)
    auc_best = roc_auc_score(y_test, y_prob_best)
    precision_score_best = np.mean(precision_best)
    recall_score_best = np.mean(recall_best)

    results.append({
        'Dataset': datasets,
        'Run': run + 1,
        'Accuracy': accuracy_best,
        'Precision': precision_score_best,
        'Recall': recall_score_best,
        'F1': f1_best,
        'MCC': mcc_best,
        'AUPR': aupr_best,
        'AUC': auc_best,
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
