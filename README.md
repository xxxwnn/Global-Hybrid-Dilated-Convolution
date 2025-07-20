# GHDC: Gut Health Deep Classifier

GHDC (Gut Health Deep Classifier) is a deep learning framework for classifying gut microbiome data using a custom 1D CNN architecture. The project supports data preprocessing, model training, evaluation, and hyperparameter optimization.

## Table of Contents

- [Features](#features)
- [Requirements](#requirements)
- [Project Structure](#project-structure)
- [Data Preparation](#data-preparation)
- [Usage](#usage)
  - [Training](#training)
  - [Evaluation](#evaluation)
  - [Hyperparameter Optimization](#hyperparameter-optimization)
- [Key Files](#key-files)
- [Citation](#citation)
- [License](#license)

---

## Features

- Custom 1D CNN model for microbiome data classification
- Data preprocessing with Centered Log-Ratio (CLR) transformation
- Support for cross-entropy and focal loss
- Model evaluation with metrics: Accuracy, AUC, F1, MCC, Precision, Recall, AUPR
- Hyperparameter optimization using Optuna
- Results logging and model checkpointing

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

## Project Structure

GHDC_Full/
├── dataloader.py # Data loading and preprocessing
├── GHDC_structure.py # CNN model definition
├── main.py # Training and evaluation script
├── objectives.py # Optuna objective for hyperparameter tuning
└── ...


## Data Preparation

- Place your Excel data files (e.g., `wt2d.xlsx`, `t2d.xlsx`, etc.) in the appropriate directory.
- The first row should contain sample labels; the rest are feature values.
- Update the `datasets` variable in `dataloader.py` to select the dataset you want to use.

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

## Key Files

- **dataloader.py**: Loads Excel data, processes labels, applies CLR transformation, and splits into train/test sets.
- **GHDC_structure.py**: Defines the CNN model architecture using PyTorch.
- **main.py**: Main training and evaluation loop, supports multiple runs and logs results.
- **objectives.py**: Contains the Optuna objective function for hyperparameter optimization and custom loss functions.

## Citation

If you use this codebase in your research, please cite appropriately.

## License

This project is licensed under the MIT License.
