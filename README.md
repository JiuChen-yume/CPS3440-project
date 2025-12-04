# CPS3440 Project

A course project for CPS3440 exploring neural-network–based regression with a focus on model accuracy, error distribution, and inference efficiency.

## 1. Overview

This repository contains code and data for training and evaluating a multilayer perceptron (MLP) model on a spatial prediction task.

The project includes:

- Data preprocessing and feature engineering  
- MLP model training and hyperparameter tuning  
- Quantitative evaluation with standard regression metrics  
- Visual analysis of prediction errors and spatial patterns  
- Comparison of inference times and model expansion behaviour  

## 2. Repository Structure

```text
CPS3440-project/
├── artifacts/          # Saved figures and result plots
├── data/
│   └── sf/             # Input data files
├── scripts/            # CLI scripts for training / evaluation
├── src/                # Core source code (models, utils, etc.)
├── README.md
└── requirements.txt    # Python dependencies
```

## 3. Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/<your-username>/CPS3440-project.git
   cd CPS3440-project
   ```

2. **Create and activate a virtual environment (optional but recommended)**

   ```bash
   python -m venv .venv
   source .venv/bin/activate   # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

## 4. Usage

### 4.1 Data Preparation

Place the required raw data files under `data/sf/`.  
If additional preprocessing is needed, use the corresponding script in `scripts/` (for example, to clean data, normalize features, or split train/test sets).

### 4.2 Training

Run the training script to fit the MLP model and save evaluation artifacts:

```bash
python scripts/train_mlp.py \
    --data_dir data/sf \
    --output_dir artifacts
```

This will:

- Train the MLP model  
- Evaluate it on the test set  
- Save plots and metrics under `artifacts/`

### 4.3 Evaluation and Inference

To run evaluation or perform inference only (using a pre-trained model):

```bash
python scripts/evaluate_mlp.py \
    --data_dir data/sf \
    --output_dir artifacts \
    --model_path artifacts/best_mlp.pt
```

## 5. Results

### 5.1 Overall MLP Metrics

Overall regression performance of the MLP model (e.g., RMSE, MAE, R²).

![MLP metrics](artifacts/mlp_metrics.png)

### 5.2 Error Bins

Binned distribution of prediction errors.

![Error bins](artifacts/error_bins.png)

### 5.3 MLP Error Coordinates

Absolute prediction error over spatial coordinates.

![MLP error coordinates](artifacts/mlp_error_coords.png)

### 5.4 MLP Error Coordinates (Difference)

Difference in error over spatial coordinates.

![MLP error coords diff](artifacts/mlp_error_coords_diff.png)

### 5.5 MLP Scatter Coordinates

Scatter plot of predictions vs. ground truth over coordinates.

![MLP scatter coordinates](artifacts/mlp_scatter_coords.png)

### 5.6 MLP Scatter Coordinates (Difference)

Difference in scatter patterns.

![MLP scatter coords diff](artifacts/mlp_scatter_coords_diff.png)

### 5.7 Expansion Bars

Expansion behaviour visualised as bar plots.

![Expansion bars](artifacts/expansion_bars.png)

### 5.8 Inference Times

Comparison of inference times for different settings.

![Inference times](artifacts/inference_times.png)

## 6. Project Notes

- All figures in the `artifacts/` directory are generated automatically by the training and evaluation scripts.  
- Random seeds may be used for reproducibility; see the relevant script arguments in `scripts/`.  
- Hyperparameters (such as learning rate, layer sizes, batch size, and number of epochs) can be configured via command-line flags or configuration files.

## 7. License

This project is for educational purposes as part of the CPS3440 course.  
If you plan to reuse or extend the code, please add an appropriate open-source license file (for example, MIT or Apache-2.0) and update this section accordingly.
