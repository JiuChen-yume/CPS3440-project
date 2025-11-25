# Shortest Path Distance Approximation using Deep Learning Techniques

本项目实现在真实道路图（OSM）上，使用图神经网络（GNN）与基线模型近似预测两个节点间的最短路径距离。包含数据抓取、特征工程、Landmark 编码、模型训练与实验评估的完整流水线。

## 项目结构

- `scripts/`
  - `generate_dataset.py`：下载城市路网并生成训练数据集（节点特征、Landmark 距离、随机节点对与真实最短路）
  - `run_experiments.py`：运行基线与模型训练，并输出评估结果
- `src/`
  - `data/`：OSM 数据抓取、预处理与工具函数
  - `baselines/`：Dijkstra 与 A* 基线实现
  - `models/`：GNN 与 MLP 模型定义
  - `training/`：模型训练脚本
  - `evaluation/`：评估指标与统一评估脚本
- `requirements.txt`：项目依赖

## 快速开始

1) 安装依赖（建议使用虚拟环境）

```
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

如果安装 `torch_geometric` 失败，请先安装 PyTorch（CPU版示例）：

```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

然后根据官方文档安装 `torch_geometric` 所需组件（Windows CPU）：
- 参考：https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html
- 或尝试：
```
pip install torch_geometric
```
（若报错请按文档安装 `torch-scatter`、`torch-sparse` 等配套轮子。）

2) 生成数据集（以旧金山为例）

```
python scripts/generate_dataset.py --place "San Francisco, California, USA" --num_pairs 20000 --landmarks 16 --outdir data/sf
```

输出内容：
- `graph.gpickle`：NetworkX 图对象
- `pairs.csv`：训练用节点对及真实最短路径距离
- `node_features.csv`：节点基础特征（经纬度、度数、PageRank 等）
- `landmark_distances.npz`：节点到每个 Landmark 的最短路径距离矩阵

3) 运行实验与评估

```
python scripts/run_experiments.py --data_dir data/sf --run_baselines --run_mlp --run_gnn
```

结果将包含：RMSE、MAE、MAPE、训练/推理耗时对比等。

## 模型与实验设计

- 节点特征：纬度、经度、度数、PageRank、是否路口(度数>2)
- Landmark 编码：选 K 个 Landmark，记录每个节点到 Landmark 的 Dijkstra 距离
- GNN 编码器：GraphSAGE K 层，获得节点嵌入
- 距离预测：取源/目标节点嵌入向量 concat，MLP 回归预测距离
- 基线：
  - Dijkstra（精确）
  - A*（启发式：地理距离）
  - MLP（仅用坐标，不用图结构）

## 评估指标

- RMSE、MAE、MAPE
- 训练耗时、推理速度
- 可选消融：不使用 Landmark、不用图结构（仅 MLP）、节点特征简化

## 典型流程与耗时建议

- 先选小区域（如旧金山）验证流程，再扩展到更大城市
- 旧金山 ~15K 节点，抓取与构图通常 30 秒内可完成
- Landmark 数量建议从 8~32 逐步调参

## 报告与分工建议（简版）

- 数据工程：OSMnx 抓取、预处理、Landmark 与样本对生成
- 模型开发：GNN/MLP 训练脚本、损失与优化
- 实验与绘图：基线对比、误差分析、耗时记录
- 报告撰写：方法与结果整理、图表与展示

## 备注

- 若 `torch_geometric` 安装困难，可先跑 MLP 与基线，记录结果与耗时；随后在可用环境下补跑 GNN。
- 所有脚本均为可读性与教学目的进行实现，生产环境可进一步优化数据加载与批处理。

