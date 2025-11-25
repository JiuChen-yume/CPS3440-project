# 基于图神经网络的最短路径距离近似研究

生成时间：2025-11-22 00:54:00

## 摘要
本文在真实道路图（旧金山）上比较 Dijkstra、A* 与 GNN（GraphSAGE）在最短路径距离预测上的性能与效率。结果显示：Dijkstra 与 A* 在精度上近乎无误差，但推理耗时显著；GNN 推理速度接近毫秒级，适合实时近似场景，但当前配置下误差较大，仍需进一步调参优化。
## 引言
最短路径计算在导航、物流与交通仿真中至关重要。随着图规模增长，传统精确算法在实时场景的耗时成为瓶颈。本文探索以 GNN 学习近似距离，实现速度与精度的权衡。
## 方法
- 基线：Dijkstra（精确）、A*（地理启发式）。
- GNN：GraphSAGE K 层编码节点嵌入，拼接源/目标嵌入经 MLP 回归距离。
- 特征工程：节点经纬度、度数、PageRank、是否路口；Landmark 距离特征。
- 评估：RMSE/MAE/MAPE 与推理耗时。
## 实验与结果
数据目录：`c:\Users\lmq\Desktop\CPS3440 project\data\sf`；综合评估：`artifacts/evaluation_summary.json`。

### 指标总览
算法 | RMSE | MAE | MAPE | 推理耗时(s)
--- | ---: | ---: | ---: | ---:
dijkstra | 0.0000 | 0.0000 | 0.0000 | 69.8178
a_star | 0.0000 | 0.0000 | 0.0000 | 28.4530
mlp | 545.9970 | 373.2797 | 0.0796 | 0.0340
gnn | 3225.5422 | 2575.7247 | 0.6357 | 0.0550

### MLP 特征消融结果表
特征 | RMSE | MAE | MAPE | 训练时长(s) | epochs | hidden_dim | lr
--- | ---: | ---: | ---: | ---: | ---: | ---: | ---:
coords | 2827.1587 | 2241.1883 | 0.4797 | 0.8736 | 10.0000 | 64.0000 | 0.0010
coords_diff | 529.0993 | 365.5667 | 0.0755 | 54.7135 | 100.0000 | 128.0000 | 0.0010

### GNN 训练与推理信息
项 | 数值
--- | ---:
RMSE | 3297.7437
MAE | 2621.1802
MAPE | 0.6204
训练时长(s) | 16.7099
推理耗时(s) | 0.0550
hidden_dim | 128.0000
num_layers | 4.0000

### 推理耗时对比
![Inference Time](figures\inference_times.png)

### MLP 指标对比与散点图
![MLP Metrics](figures\mlp_metrics.png)
![MLP Scatter](figures\mlp_scatter_coords_diff.png)
![MLP Error](figures\mlp_error_coords_diff.png)

### 分距离段误差分箱分析
![Error Bins](figures\error_bins.png)

## 理论与启发式分析
- A* 启发式使用地理直线距离（Haversine），在边权为几何距离且满足三角不等式的道路网络上是可采纳且一致的：不会高估真实最短距离，且满足 `h(u) ≤ w(u,v)+h(v)`，因此 A* 保证找到最优路径并能显著减少扩展节点数。
- 相比 Dijkstra，A* 在具有空间嵌入的图（如道路网络）能以常数因子加速，结合分箱误差图与扩展计数对比，可见在不同距离段均能缩小搜索范围。

### 扩展节点与前沿大小对比（A* vs Dijkstra）
项 | 数值
--- | ---:
样本对数 | 200.0000
Dijkstra平均扩展节点 | 5765.5800
A*平均扩展节点 | 1431.7200
Dijkstra平均前沿大小 | 161.1800
A*平均前沿大小 | 158.1400
扩展速度提升倍数 | 4.0270
![Expansion Bars](figures\expansion_bars.png)

## 讨论
- 速度与精度：GNN 在毫秒级推理上有明显优势；现有配置下误差较高，需调参与特征增强。
- 调参方向：增加训练轮数/隐藏维度、使用 HuberLoss、log1p 目标、增加 Landmark、改进特征。
- 工程鲁棒性：对 `NaN/Inf` 特征与不可达样本做清洗，避免训练与评估阶段的数值异常。

## 结论
在真实道路网络上，GNN 近似最短路径距离的速度优势显著，但需要进一步优化以达到实用精度；Dijkstra/A* 仍是精确参考与上限基线。

## 参考文献
[1] Hamilton et al., Inductive Representation Learning on Large Graphs (GraphSAGE).
[2] Dijkstra, A note on two problems in connexion with graphs, 1959.
[3] Hart et al., A Formal Basis for the Heuristic Determination of Minimum Cost Paths, 1968.