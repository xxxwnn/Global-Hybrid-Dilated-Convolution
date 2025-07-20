import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.stats.power import TTestIndPower
from statsmodels.stats.multitest import multipletests  # 导入FDR矫正函数

# 读取数据（适用于 CSV，也支持 Excel 只需换 read_excel）
datasets = "crucial germs wt2d"
df = pd.read_excel(fr'C:\Users\xxwn\Desktop\bio\Gut_flora_v1\data1\{datasets}.xlsx', header=None)

# 提取标签和数据
labels = df.iloc[0, 1:].values  # 第一行是标签，从第2列开始是样本
otu_names = df.iloc[1:, 0].values  # 第一列是OTU名字，从第2行开始是数据
data = df.iloc[1:, 1:].astype(float)  # 去掉标签和 OTU ID，转为 float 类型

# 删除全为0的行
print(f"原始数据形状: {data.shape}")
zero_rows = (data == 0).all(axis=1)
data = data[~zero_rows]
otu_names = otu_names[~zero_rows]
print(f"删除全为0的行后数据形状: {data.shape}")
print(f"删除了 {zero_rows.sum()} 个全为0的OTU")

# 将数据分组
group_healthy = data.loc[:, labels == 'n'].values.T  # 健康组
group_disease = data.loc[:, labels != 'n'].values.T  # 患病组

# 准备统计输出
results = []

# 分析每一个 OTU（行）
for i in range(data.shape[0]):
    otu_id = otu_names[i]
    x = group_healthy[:, i]
    y = group_disease[:, i]

    # 方差齐性检验
    levene_p = stats.levene(x, y).pvalue
    equal_var = levene_p > 0.05

    # t检验
    t_stat, p_val = stats.ttest_ind(x, y, equal_var=equal_var)

    # 均值
    mean_x = np.mean(x)
    mean_y = np.mean(y)

    # Cohen's d
    nx, ny = len(x), len(y)
    pooled_std = np.sqrt(((nx - 1) * np.var(x, ddof=1) + (ny - 1) * np.var(y, ddof=1)) / (nx + ny - 2))
    d = (mean_x - mean_y) / pooled_std if pooled_std != 0 else 0

    # 功效分析
    try:
        analysis = TTestIndPower()
        power = analysis.power(effect_size=abs(d), nobs1=nx, ratio=ny / nx, alpha=0.05)
    except:
        power = np.nan

    results.append({
        'OTU_ID': otu_id,
        'p-value': p_val,
        't-value': t_stat,
        'Cohen_d': d,
        'Power': power,
        'Healthy_mean': mean_x,
        'Disease_mean': mean_y,
        'Equal_Variance': equal_var
    })

# 将结果转为DataFrame
result_df = pd.DataFrame(results)

# FDR矫正（Benjamini-Hochberg方法）
fdr_results = multipletests(result_df['p-value'], alpha=0.05, method='fdr_bh')
result_df['FDR_q-value'] = fdr_results[1]  # 添加FDR校正后的q值
result_df['FDR_significant'] = fdr_results[0]  # 添加是否显著的布尔值

# 按FDR q-value排序
result_df = result_df.sort_values(by='FDR_q-value')

# 保存结果
result_df.to_csv(fr"C:\Users\xxwn\Desktop\bio\Gut_flora_v1\power_analysis\crucial germs\otu_ttest_results_{datasets}.csv", index=False)

print("分析完成，结果保存在 'otu_ttest_results_with_fdr.csv'")