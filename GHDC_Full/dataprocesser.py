import pandas as pd
import numpy as np
import os

dataset = 'wt2d'
# 读取数据
cir = pd.read_csv(fr'C:\Users\xxwn\Desktop\bio\Gut_flora_v1\data1\{dataset}.csv', header=None)

rows_to_drop = []
num_rows, num_cols = cir.shape

# 你可以根据实际情况调整这个阈值
large_dataset_threshold = 200

for index, row in cir.iterrows():
    zero_count = (row == 0).sum() + (row == '0').sum()
    if num_cols > large_dataset_threshold:
        # 大数据集：零值比例不能超过30%
        if zero_count > num_cols * 0.3:
            rows_to_drop.append(index)
    else:
        # 小数据集：零值数量不能超过40
        if zero_count > 25:
            rows_to_drop.append(index)

cir = cir.drop(rows_to_drop, axis=0)
os.makedirs(r"C:\Users\xxwn\Desktop\bio\Gut_flora_v1\data.25", exist_ok=True)
cir.to_excel(fr"C:\Users\xxwn\Desktop\bio\Gut_flora_v1\data.25\{dataset}.xlsx", header=None)
print(cir)
