import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, matthews_corrcoef, precision_recall_curve, auc, \
    roc_auc_score, f1_score
from sklearn.utils import shuffle
import optuna
import joblib

from dataloader import x, y, datasets

model_path = fr'C:\Users\xxwn\Desktop\bio\Gut_flora_v1\RF\{datasets}\pth\best_model_{datasets}_crucial.pth'
#
# # 加入clr归一化
# def clr_transform(X, eps=1e-8):
#     import numpy as np
#     X = np.array(X)
#     X = np.where(X <= 0, eps, X)  # 防止log(0)
#     log_X = np.log(X)
#     mean_log = np.mean(log_X, axis=1, keepdims=True)
#     clr_X = log_X - mean_log
#     return clr_X
#
#
# x = clr_transform(x)
#
# # 数据集划分
# X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
#
#
# def objective(trial):
#     X_train_shuffled, y_train_shuffled = shuffle(X_train, y_train, random_state=trial.number)
#     # Optuna 参数调优
#     n_estimators = trial.suggest_int("n_estimators", 50, 300)
#     max_depth = trial.suggest_int("max_depth", 5, 30)
#     min_samples_split = trial.suggest_int("min_samples_split", 2, 20)
#
#     rf_model = RandomForestClassifier(
#         n_estimators=n_estimators,
#         max_depth=max_depth,
#         min_samples_split=min_samples_split,
#         random_state=42
#     )
#
#     rf_model.fit(X_train_shuffled, y_train_shuffled)
#     y_pred = rf_model.predict(X_test)
#
#     accuracy = accuracy_score(y_test, y_pred)
#     mcc = matthews_corrcoef(y_test, y_pred)
#
#     y_prob = rf_model.predict_proba(X_test)[:, 1]
#     precision, recall, _ = precision_recall_curve(y_test, y_prob)
#     auc1 = roc_auc_score(y_test, y_prob)
#     aupr = auc(recall, precision)
#
#     print(f"Trial {trial.number}: Accuracy: {accuracy:.4f}, MCC: {mcc:.4f}, AUPR: {aupr:.4f}")
#     return auc1
#
#
# study = optuna.create_study(direction="maximize")
# study.optimize(objective, n_trials=30)
#
# best_params = study.best_params
# print("Best parameters found by Optuna:", best_params)
#
# rf_best_model = RandomForestClassifier(
#     n_estimators=best_params["n_estimators"],
#     max_depth=best_params["max_depth"],
#     min_samples_split=best_params["min_samples_split"],
#     random_state=42
# )
#
# rf_best_model.fit(X_train, y_train)
# y_pred_best = rf_best_model.predict(X_test)
#
# # # 保存模型
# # joblib.dump(rf_best_model, model_path)
# # print(f"模型已保存到 {model_path}")
#
# accuracy_best = accuracy_score(y_test, y_pred_best)
# mcc_best = matthews_corrcoef(y_test, y_pred_best)
#
# y_prob_best = rf_best_model.predict_proba(X_test)[:, 1]
# precision_best, recall_best, _ = precision_recall_curve(y_test, y_prob_best)
# aupr_best = auc(recall_best, precision_best)
#
# print(f"Best Model Accuracy: {accuracy_best:.4f}")
# print(f"Best Model MCC: {mcc_best:.4f}")
# print(f"Best Model AUPR: {aupr_best:.4f}")
# print("Classification Report:\n", classification_report(y_test, y_pred_best))
#
# # 计算并输出AUC分数
# auc_best = roc_auc_score(y_test, y_prob_best)
# print(f"Best Model AUC: {auc_best:.4f}")
#
# test_accuracies = [trial.value for trial in study.trials]
# plt.plot(range(1, len(test_accuracies) + 1), test_accuracies, marker='o')
# plt.xlabel('Trial')
# plt.ylabel('Accuracy')
# plt.title('Optuna Trials - Accuracy')
# plt.grid(True)
# plt.show()
import matplotlib.pyplot as plt


# from sklearn.model_selection import train_test_split, KFold
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score, classification_report, matthews_corrcoef, precision_recall_curve, auc, roc_auc_score
# from sklearn.utils import shuffle
# import optuna
# import joblib
# import pandas as pd
# import numpy as np
# from dataloader import x, y, model_path

# CLR transformation
def clr_transform(X, eps=1e-8):
    X = np.array(X)
    X = np.where(X <= 0, eps, X)  # Prevent log(0)
    log_X = np.log(X)
    mean_log = np.mean(log_X, axis=1, keepdims=True)
    clr_X = log_X - mean_log
    return clr_X


x = clr_transform(x)


# Define Optuna objective function
def objective(trial):
    X_train_shuffled, y_train_shuffled = shuffle(X_train, y_train, random_state=trial.number)
    n_estimators = trial.suggest_int("n_estimators", 50, 300)
    max_depth = trial.suggest_int("max_depth", 5, 30)
    min_samples_split = trial.suggest_int("min_samples_split", 2, 20)

    rf_model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        random_state=42
    )

    rf_model.fit(X_train_shuffled, y_train_shuffled)
    y_pred = rf_model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)
    y_prob = rf_model.predict_proba(X_test)[:, 1]
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    auc1 = roc_auc_score(y_test, y_prob)
    aupr = auc(recall, precision)

    print(f"Trial {trial.number}: Accuracy: {accuracy:.4f}, MCC: {mcc:.4f}, AUPR: {aupr:.4f}")
    return auc1


# Run Optuna optimization
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
y = np.array(y)
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=15)

best_params = study.best_params
print("Best parameters found by Optuna:", best_params)

# Perform 10-fold cross-validation with best parameters
kf = KFold(n_splits=10, shuffle=True, random_state=42)
results = []

for fold, (train_idx, test_idx) in enumerate(kf.split(x)):
    X_train_fold, X_test_fold = x[train_idx], x[test_idx]
    y_train_fold, y_test_fold = y[train_idx], y[test_idx]

    # 在每个fold内定义Optuna目标函数
    def fold_objective(trial):
        X_train_shuffled, y_train_shuffled = shuffle(X_train_fold, y_train_fold, random_state=trial.number)
        n_estimators = trial.suggest_int("n_estimators", 50, 300)
        max_depth = trial.suggest_int("max_depth", 5, 30)
        min_samples_split = trial.suggest_int("min_samples_split", 2, 20)

        rf_model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=42
        )

        rf_model.fit(X_train_shuffled, y_train_shuffled)
        y_pred = rf_model.predict(X_test_fold)
        y_prob = rf_model.predict_proba(X_test_fold)[:, 1]
        auc1 = roc_auc_score(y_test_fold, y_prob)
        return auc1

    # 每个fold都新建study
    fold_study = optuna.create_study(direction="maximize")
    fold_study.optimize(fold_objective, n_trials=15)
    fold_best_params = fold_study.best_params

    # 用最优参数训练模型
    rf_best_model = RandomForestClassifier(
        n_estimators=fold_best_params["n_estimators"],
        max_depth=fold_best_params["max_depth"],
        min_samples_split=fold_best_params["min_samples_split"],
        random_state=42
    )
    rf_best_model.fit(X_train_fold, y_train_fold)
    y_pred_fold = rf_best_model.predict(X_test_fold)
    y_prob_fold = rf_best_model.predict_proba(X_test_fold)[:, 1]

    # 计算指标
    accuracy = accuracy_score(y_test_fold, y_pred_fold)
    mcc = matthews_corrcoef(y_test_fold, y_pred_fold)
    precision, recall, _ = precision_recall_curve(y_test_fold, y_prob_fold)
    aupr = auc(recall, precision)
    auc_score = roc_auc_score(y_test_fold, y_prob_fold)
    f1 = f1_score(y_test_fold, y_pred_fold)

    # 存储结果，加入数据集名称
    results.append({
        'Dataset': datasets,
        'Fold': fold + 1,
        'Accuracy': accuracy,
        'Precision': np.mean(precision),
        'Recall': np.mean(recall),
        'F1': f1,
        'MCC': mcc,
        'AUPR': aupr,
        'AUC': auc_score,
        'Best_Params': str(fold_best_params)
    })

    print(f"Fold {fold + 1}: Accuracy: {accuracy:.4f}, MCC: {mcc:.4f}, AUPR: {aupr:.4f}, AUC: {auc_score:.4f}")

# Save results to Excel
results_df = pd.DataFrame(results)
excel_path = r'C:\Users\xxwn\Desktop\bio\Gut_flora_v1\RF\model_performance.xlsx'

if os.path.exists(excel_path):
    # 读取原有内容，合并后写回
    old_df = pd.read_excel(excel_path)
    all_df = pd.concat([old_df, results_df], ignore_index=True)
    all_df.to_excel(excel_path, index=False)
else:
    results_df.to_excel(excel_path, index=False)

print("Results saved to model_performance.xlsx")

# Save the final model trained on the original split
rf_best_model = RandomForestClassifier(
    n_estimators=best_params["n_estimators"],
    max_depth=best_params["max_depth"],
    min_samples_split=best_params["min_samples_split"],
    random_state=42
)
rf_best_model.fit(X_train, y_train)
os.makedirs(fr'C:\Users\xxwn\Desktop\bio\Gut_flora_v1\RF\{datasets}\pth')
joblib.dump(rf_best_model, model_path)
print(f"Final model saved to {model_path}")
