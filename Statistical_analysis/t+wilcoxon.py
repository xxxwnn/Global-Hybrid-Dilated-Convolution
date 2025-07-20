import pandas as pd
import os
import glob
from scipy.stats import ttest_rel, wilcoxon
from statsmodels.stats.multitest import multipletests

# the directory of results of all models
data_dir = r"C:\Users\xxwn\Desktop\bio\Gut_flora_v1\Statistical_analysis"
metric = "AUPR"
model_data = {}

for file in glob.glob(os.path.join(data_dir, "*.csv")):
    model_name = os.path.splitext(os.path.basename(file))[0].lower()
    df = pd.read_csv(file)
    df = df[["dataset", "run_id", metric]].copy()
    df.rename(columns={metric: model_name}, inplace=True)
    model_data[model_name] = df

for file in glob.glob(os.path.join(data_dir, "*.xlsx")):
    model_name = os.path.splitext(os.path.basename(file))[0].lower()
    df = pd.read_excel(file)
    df = df[["dataset", "run_id", metric]].copy()
    df.rename(columns={metric: model_name}, inplace=True)
    model_data[model_name] = df


from functools import reduce
dfs = list(model_data.values())
merged = reduce(lambda left, right: pd.merge(left, right, on=["dataset", "run_id"]), dfs)

print(merged.head())

# the model aiming to analysis
main_model = "model_performance_ghdc"
results = []
datasets = merged["dataset"].unique()
for dataset in datasets:
    subset = merged[merged["dataset"] == dataset]
    for model in merged.columns[2:]:
        if model == main_model:
            continue
        ghdc_scores = subset[main_model].values
        other_scores = subset[model].values

        # t-examination
        t_stat, p_ttest = ttest_rel(ghdc_scores, other_scores)

        # Wilcoxon examination
        try:
            w_stat, p_wilcoxon = wilcoxon(ghdc_scores - other_scores)
        except ValueError:  # all zeros
            p_wilcoxon = 1.0

        results.append({
            "Dataset": dataset,
            "Model": model,
            "Paired t-test p": p_ttest,
            "Wilcoxon p": p_wilcoxon,
            "Mean Δ": (ghdc_scores - other_scores).mean()
        })

result_df = pd.DataFrame(results)

# FDR
for test in ["Paired t-test p", "Wilcoxon p"]:
    _, pvals_corrected, _, _ = multipletests(result_df[test], alpha=0.05, method="fdr_bh")
    result_df[f"{test} (FDR adj)"] = pvals_corrected
    result_df[f"{test} sig"] = pvals_corrected < 0.05

print(result_df.to_string(index=False))

# 可选：保存结果为 Excel
result_df.to_excel(fr"./res/ghdc_statistical_comparison_{metric}.xlsx", index=False)
