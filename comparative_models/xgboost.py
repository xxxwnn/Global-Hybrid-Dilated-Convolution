import numpy as np
import xgboost as xgb
import optuna
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef, \
    average_precision_score

# 数据集名称
datasets = 'wt2d'  # 可根据实际情况修改
excel_path = r'C:\Users\xxwn\Desktop\bio\Gut_flora_v1\XGBoost\xgb_performance.xlsx'  # 结果保存路径


# CLR变换
def clr_transform(X, eps=1e-8):
    X = np.array(X)
    X = np.where(X <= 0, eps, X)
    log_X = np.log(X)
    mean_log = np.mean(log_X, axis=1, keepdims=True)
    clr_X = log_X - mean_log
    return clr_X


# 读取数据
dt = pd.read_excel(fr'C:\Users\xxwn\Desktop\bio\Gut_flora_v1\data2\{datasets}.xlsx', header=None)
labels = dt.iloc[0, 1:].apply(lambda x: 0 if 'n' in str(x).lower() else 1).values
features = dt.iloc[1:, 1:].values.astype(float)
features = clr_transform(features)
features = features.T  # 保证样本为行

results = []

for run in range(10):
    print(f'========== Run {run + 1} ==========')
    # 1. 先划分20%为hold-out测试集，80%为训练集
    X_trainval, X_test, y_trainval, y_test = train_test_split(features, labels, test_size=0.2, random_state=run,
                                                              stratify=labels)
    # 2. 在80%训练集上再划分出验证集（比如20%验证，80%训练），用于Optuna调参
    X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, test_size=0.2, random_state=run,
                                                      stratify=y_trainval)

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    dtrainval = xgb.DMatrix(X_trainval, label=y_trainval)
    dtest = xgb.DMatrix(X_test, label=y_test)


    # 3. Optuna调参（只用80%训练集和其中的验证集）
    def objective(trial):
        params = {
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 1.0),
            'n_estimators': trial.suggest_int('n_estimators', 50, 500),
            'subsample': trial.suggest_uniform('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.5, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        }
        # 用早停防止过拟合
        model = xgb.train(params, dtrain, num_boost_round=100, early_stopping_rounds=10,
                          evals=[(dtrain, 'train'), (dval, 'validation')],
                          verbose_eval=False)
        y_pred_proba = model.predict(dval)
        auc = roc_auc_score(y_val, y_pred_proba)
        return auc


    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=15, show_progress_bar=False)
    best_params = study.best_params

    # 4. 用最优参数在80%训练集（trainval）上训练最终模型
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'max_depth': best_params['max_depth'],
        'eta': best_params['learning_rate'],
        # 'n_estimators': best_params['n_estimators'],
        'subsample': best_params['subsample'],
        'colsample_bytree': best_params['colsample_bytree'],
        'min_child_weight': best_params['min_child_weight'],
    }
    final_model = xgb.train(params, dtrainval, num_boost_round=100, evals=[(dtrainval, 'train')],
                            early_stopping_rounds=10, verbose_eval=False)

    # 5. 在20% hold-out测试集上评估
    y_pred_proba = final_model.predict(dtest)
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
