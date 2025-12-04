Graph-Based Route Distance Prediction with MLP & GNN

A CPS3440 Course Project

This project implements and compares multiple methods for predicting shortest-path distances in the San Francisco road network (SF graph).
It includes:

Exact routing algorithms (Dijkstra, A*)

Machine learning baselines (MLP)

Graph Neural Networks (GNN)

Full evaluation suite

Visualization pipeline (error maps, scatter plots, metrics, inference time, node expansions)

The project is fully reproducible and organized as a modular research pipeline.


1. Project Structure

 project/
│
├── data/
│   └── sf/
│       ├── graph.gpickle
│       ├── node_features.csv
│       ├── pairs.csv
│       ├── landmark_distances.npz
│       └── artifacts/              # generated results, models, plots as png files (only uploaded this fold)
│
├── src/
│   ├── data/                       # data loaders
│   ├── evaluation/                 # metrics & evaluation summary
│   ├── models/                     # GNN & MLP model definitions
│   ├── training/                   # training loops
│   └── utils/                      # helper functions
│
└── scripts/
    ├── run_experiments.py          # main experiment runner
    ├── generate_dataset.py
    ├── analyze_error_bins.py
    ├── expansion_analysis.py
    ├── plot_results.py             # generate all plots
    └── render_report.py


2. Installation

Create virtual environment

python -m venv .venv
.\.venv\Scripts\activate  

Install dependencies

pip install -r requirements.txt

3. Running Experiments

(1) Run classical algorithms (baseline)

python scripts/run_experiments.py --data_dir data/sf --run_baselines

(2) Train MLP model

python scripts/run_experiments.py --data_dir data/sf --run_mlp

(3) Train GNN model

python scripts/run_experiments.py --data_dir data/sf --run_gnn

(4) Generate all plots

python scripts/run_experiments.py --data_dir data/sf --gen_plots

All outputs will appear in: data/sf/artifacts/

4. Results

Below are key results generated from the dataset.  
All plots are stored in the artifacts/ folder of the repository.

4.1 MLP Metrics Across Feature Sets  
This plot compares MLP performance using two types of input features:
- coords (raw coordinates)
- coords_diff (coordinate differences)

The coords_diff version dramatically improves RMSE and MAE.

![MLP metrics](artifacts/mlp_metrics.png)


4.2 Algorithm Inference Time  
This figure compares running time of:
- Dijkstra
- A*
- MLP
- GNN

MLP and GNN achieve millisecond-level inference, much faster than classical routing algorithms.

![Inference times](artifacts/inference_times.png)


4.3 Error by Distance Bins  
This figure shows prediction error (MAE) grouped by true distance range.

Main observations:
- GNN error increases with distance
- MLP (coords_diff) stays relatively stable

![Error bins](artifacts/error_bins.png)


4.4 MLP Error vs True Distance (coords)  
Using raw coords as features leads to large systematic bias in the prediction errors.

![MLP error coords](artifacts/mlp_error_coords.png)


4.5 MLP Error vs True Distance (coords_diff)  
Using coordinate differences instead of raw coordinates reduces both bias and variance in the errors.

![MLP error coords diff](artifacts/mlp_error_coords_diff.png)


4.6 Prediction Scatter Plot (coords)  
Scatter plot of true distance versus predicted distance using raw coords.  
Predictions deviate significantly from the ideal diagonal, indicating underestimation and bias.

![MLP scatter coords](artifacts/mlp_scatter_coords.png)


4.7 Prediction Scatter Plot (coords_diff)  
Scatter plot of true distance versus predicted distance using coords_diff.  
The points align much more closely with the diagonal, showing strong improvement in model quality.

![MLP scatter coords diff](artifacts/mlp_scatter_coords_diff.png)


4.8 Node Expansions: Dijkstra vs A*  
This bar chart shows the average number of nodes expanded and frontier sizes for Dijkstra and A*.  
A* expands far fewer nodes than Dijkstra, which confirms its higher efficiency on this road network.

![Node expansions](artifacts/expansion_bars.png)

5. Key Findings

- A* is significantly faster than Dijkstra for shortest-path estimation.  
- MLP using coords_diff provides the best trade-off between accuracy and computation time.  
- GNN performs reasonably well but struggles with longer distances.  
- Feature engineering has a larger impact than model choice.  
- Machine learning models offer millisecond-level inference time, suitable for large-scale applications.  
- Hybrid approaches combining ML prediction with classical search may offer further improvements.


6. Team Members
   










