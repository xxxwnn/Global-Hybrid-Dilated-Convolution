import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import colormaps
from matplotlib.colors import Normalize

# Set Seaborn style for better aesthetics
sns.set_style("whitegrid", {"grid.color": ".85", "grid.linestyle": ":"})
dataset = 't2d'
# Load the data
df = pd.read_csv(fr"C:\Users\xxwn\Desktop\bio\Gut_flora_v1\power_analysis\cut zeros\otu_ttest_results_{dataset}.csv")

# Calculate log2 fold change (差异倍数)
df['log2_fold_change'] = np.log2(df['Disease_mean'] / df['Healthy_mean'].replace(0, 1e-10))

# Calculate -log10(FDR_q-value), replacing 0 with a small value to avoid log(0)
df['neg_log10_fdr'] = -np.log10(df['FDR_q-value'].replace(0, 1e-10))

# Map Power values to colors using viridis colormap
norm = Normalize(vmin=df['Power'].min(), vmax=df['Power'].max())
cmap = colormaps['viridis']  # Updated to use matplotlib.colormaps
df['power_color'] = [cmap(norm(x)) for x in df['Power']]  # Convert Power to colors

# Initialize the plot
plt.figure(figsize=(12, 8), dpi=300)

# Scatter plot with color coding for significance and power
scatter = sns.scatterplot(
    x='log2_fold_change',
    y='neg_log10_fdr',
    hue='FDR_significant',
    size=5,  # Fixed size for clarity
    palette={True: '#e41a1c', False: '#999999'},  # Red for significant, gray for non-significant
    style='FDR_significant',  # Different markers for significant vs non-significant
    markers={True: 'o', False: '^'},  # Circle for significant, triangle for non-significant
    alpha=0.7,
    edgecolor='black',
    linewidth=0.2,
    data=df
)

# Add significance threshold line (FDR_q-value = 0.05)
plt.axhline(y=-np.log10(0.05), color='navy', linestyle='--', linewidth=1.5, label='FDR = 0.05')

# Add vertical lines for fold change thresholds (差异倍数阈值线)
plt.axvline(x=1, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='log2 FC = 1 (2倍)')
plt.axvline(x=-1, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='log2 FC = -1 (0.5倍)')
plt.axvline(x=0, color='black', linestyle='--', linewidth=1, alpha=0.3, label='log2 FC = 0 (无变化)')

# Customize axes
plt.xlabel("log2 Fold Change", fontsize=14, fontweight='bold')
plt.ylabel("-log10(FDR q-value)", fontsize=14, fontweight='bold')
plt.title(f"Volcano Plot of OTU Differential Abundance ({dataset} vs. Healthy)", fontsize=16, pad=15)

# Add annotations for top 10 significant OTUs
top_significant = df[df['FDR_significant']].sort_values('FDR_q-value').head(10)
for i, row in top_significant.iterrows():
    plt.text(
        row['log2_fold_change'] + (0.1 if row['log2_fold_change'] > 0 else -0.1),  # Slight offset for readability
        row['neg_log10_fdr'],
        row['OTU_ID'].split('|')[-1],  # Show only species name
        fontsize=9,
        ha='left' if row['log2_fold_change'] < 0 else 'right',
        va='center',
        bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=1)
    )

# Customize legend
plt.legend(
    title='FDR Significant',
    loc='upper center',
    fontsize=10,
    title_fontsize=12,
    labels=['Non-significant', 'Significant (q < 0.05)'],
    handles=[
        plt.scatter([], [], c='#999999', marker='^', label='Non-significant', edgecolor='black', s=50),
        plt.scatter([], [], c='#e41a1c', marker='o', label='Significant', edgecolor='black', s=50)
    ]
)

# Add colorbar for power
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
cbar = plt.colorbar(sm, ax=plt.gca(), label='Statistical Power', pad=0.02)
cbar.ax.tick_params(labelsize=10)
cbar.set_label('Statistical Power', fontsize=12, fontweight='bold')

# Adjust layout for tightness
plt.tight_layout()

# Save and show the plot
plt.savefig(fr"C:\Users\xxwn\Desktop\bio\Gut_flora_v1\power_analysis\cut zeros"
            fr"\volcano_plot_{dataset}_optimized_fixed.png", dpi=300, bbox_inches='tight', format='png')
# plt.show()