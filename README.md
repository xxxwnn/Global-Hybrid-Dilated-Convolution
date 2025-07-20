# GHDC: Global Hybird Dilated Convolution

GHDC (Global Hybird Dilated Convolution) is a deep learning framework for disesase prediction using a customnized 1D Dilated-CNN architecture. The project supports data preprocessing, model training, evaluation, and hyperparameter optimization.

## Table of Contents

- [Requirements](#requirements)
- [Data Preparation & Preprocessing](#data-preparation--preprocessing)
- [Usage](#usage)
  - [Training](#training)
  - [Evaluation](#evaluation)
  - [Hyperparameter Optimization](#hyperparameter-optimization)
  - [Statistical Analysis](#statistical-analysis)
  - [Power Analysis](#power-analysis)
- [Key Files](#key-files)
- [Citation](#citation)

---

## Requirements

- Python 3.7+
- PyTorch
- scikit-learn
- pandas
- numpy
- optuna
- matplotlib
- tensorboard

Install dependencies:

```bash
pip install -r requirements.txt
```

## Data Preparation & Preprocessing

### Data Type

- **Input Data:** Tabular microbiome abundance data in `.xlsx` or `.csv` format.
- **Rows:** Features (e.g., OTUs, species, or genes).
- **Columns:** Samples. The first row contains sample labels (e.g., case/control), and the rest are feature values.

### Preprocessing Steps

1. **Zero-Value Filtering (`dataprocesser.py`):**
   - For each feature (row), count the number of zero values.
   - For large datasets (columns > 200): remove features with more than 30% zero values.
   - For small datasets: remove features with more than 40 zero values.
   - The cleaned data is saved as an Excel file for downstream analysis.

2. **Label Extraction & CLR Transformation (`dataloader.py`):**
   - Extracts numeric labels from the first row (e.g., 0 for normal, 1 for disease).
   - Drops the first row and first column to keep only feature values.
   - Applies Centered Log-Ratio (CLR) transformation to the feature matrix to normalize compositional data:
     - Adds a small constant to avoid log(0).
     - Computes the geometric mean for each sample.
     - Applies `log(x_i / geometric_mean)` for each feature.
   - Splits the data into training and test sets.

### Example: Data Cleaning

```python
# dataprocesser.py
import pandas as pd
import numpy as np
import os

dataset = 'wt2d'
cir = pd.read_csv(fr'C:\path\to\{dataset}.csv', header=None)

rows_to_drop = []
num_rows, num_cols = cir.shape
large_dataset_threshold = 200

for index, row in cir.iterrows():
    zero_count = (row == 0).sum() + (row == '0').sum()
    if num_cols > large_dataset_threshold:
        if zero_count > num_cols * 0.3:
            rows_to_drop.append(index)
    else:
        if zero_count > 40:
            rows_to_drop.append(index)

cir = cir.drop(rows_to_drop, axis=0)
os.makedirs(r"C:\path\to\cleaned_data", exist_ok=True)
cir.to_excel(fr"C:\path\to\cleaned_data\{dataset}.xlsx", header=None)
```

### Example: CLR Transformation

```python
# dataloader.py (excerpt)
data = np.array(data, dtype=np.float32)
data = data + 1e-8  # Avoid log(0)
geometric_mean = np.exp(np.mean(np.log(data), axis=1, keepdims=True))
clr_data = np.log(data / geometric_mean)
```

## Usage

### Training

To train the GHDC model:

```bash
python main.py
```

- The script will automatically split the data, train the model for multiple runs, and save results to `metrics_summary_runs.csv`.
- Training uses GPU if available.

### Evaluation

- During training, the script evaluates the model on a held-out test set after each epoch.
- Metrics such as Accuracy, AUC, F1, MCC, Precision, Recall, and AUPR are computed and logged.

### Hyperparameter Optimization

To perform hyperparameter tuning with Optuna:

1. Edit and run the relevant code in `objectives.py`.
2. The `objective` function defines the search space and training loop.
3. Best models are saved automatically.

### Statistical Analysis

To compare the performance of GHDC and other models across datasets using paired t-test and Wilcoxon signed-rank test:

```bash
python Statistical_analysis/t+wilcoxon.py
```

- The script merges performance metrics (e.g., AUPR) from multiple model result files (CSV/XLSX).
- For each dataset and model, it computes:
  - Paired t-test p-value
  - Wilcoxon signed-rank test p-value
  - FDR-adjusted p-values (Benjamini-Hochberg)
  - Mean performance difference
- Results are saved as an Excel file for further inspection.

### Power Analysis

To perform power analysis and feature-level statistical testing (t-test, Cohen's d, FDR correction):

```bash
python power_analysis/power_analysis.py
```

- The script reads OTU abundance data and sample labels.
- For each OTU (feature), it:
  - Removes features with all zero values.
  - Splits samples into healthy and disease groups.
  - Performs variance homogeneity test, t-test, computes Cohen's d, and statistical power.
  - Applies FDR correction to p-values.
- Results are saved as a CSV file, sorted by FDR q-value.

## Key Files

- **dataloader.py**: Loads Excel data, processes labels, applies CLR transformation, and splits into train/test sets.
- **GHDC_structure.py**: Defines the CNN model architecture using PyTorch.
- **main.py**: Main training and evaluation loop, supports multiple runs and logs results.
- **objectives.py**: Contains the Optuna objective function for hyperparameter optimization and custom loss functions.

## Citation

If you use this codebase in your research, please cite appropriately.
