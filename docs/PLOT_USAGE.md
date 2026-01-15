# 基因相关性可视化使用指南

本指南介绍如何使用 `gene_correlation_plot.py` 模块进行相关性分析结果的可视化。

## 功能概述

`CorrelationVisualizer` 类提供以下可视化功能：

1. **单基因对散点图** - 展示两个基因之间的相关性
2. **Top N 基因对网格图** - 批量展示最相关的基因对
3. **多基因对热图** - 矩阵式展示多个基因对的相关性
4. **完整相关性矩阵热图** - 展示所有基因的相关性矩阵

## 基本用法

### 初始化可视化器

```python
from gene_correlation_ultra.gene_correlation_plot import CorrelationVisualizer

visualizer = CorrelationVisualizer(
    st_expr_matrix=adata,              # AnnData 对象
    output_dir="correlation_plots",    # 输出目录
    dpi=300,                           # 图片分辨率
    figsize=(8, 8)                     # 默认图片大小
)
```

### 1. 单基因对散点图

展示两个基因之间的相关性和散点分布。

```python
visualizer.plot_single_pair_scatter(
    gene_x="EGFR",                     # X轴基因
    gene_y="KRAS",                     # Y轴基因
    corr=0.85,                         # 相关系数（可选）
    pvalue=0.001,                      # P值（可选）
    figsize=(10, 10),                  # 图片大小
    fontsize=12,                       # 字体大小
    show_plot=True                     # 是否显示图片
)
```

输出：`scatter_EGFR_KRAS.png`

### 2. Top N 基因对网格图

批量展示相关性最强的基因对。

```python
visualizer.plot_top_pairs_scatter_grid(
    sig_pairs=sig_pairs,              # 显著相关对 DataFrame
    top_n=12,                          # 展示前 N 个基因对
    n_cols=4,                          # 每列 4 个图
    figsize=(20, 25),                  # 总图片大小
    fontsize=10
)
```

输出：`scatter_grid_top12.png`

**参数说明**：
- `sig_pairs`: 必须包含 `gene_x`, `gene_y`, `correlation`, `p_value` 列
- `top_n`: 默认 12
- `n_cols`: 根据图片大小自动计算，建议 3-4

### 3. 多基因对热图

矩阵式展示多个基因对的相关性。

```python
# 定义基因对列表
gene_pairs = [
    ("EGFR", "KRAS"),
    ("TP53", "MDM2"),
    ("BRAF", "MEK1")
]

visualizer.plot_multiple_pairs_heatmap(
    gene_pairs=gene_pairs,             # 基因对列表
    sig_pairs=sig_pairs,              # 显著相关对数据
    figsize=(12, 10),                 # 图片大小
    fontsize=12,
    cmap="RdBu_r",                    # 颜色映射
    vmin=-1,                          # 最小值
    vmax=1                            # 最大值
)
```

输出：`heatmap_gene_pairs.png`

**颜色映射选项**：
- `RdBu_r`: 红蓝渐变（推荐）
- `coolwarm`: 冷暖色
- `viridis`: 黄绿紫
- `plasma`: 紫黄

### 4. 完整相关性矩阵热图

展示所有基因的完整相关性矩阵。

```python
visualizer.plot_correlation_matrix_heatmap(
    gene_list=["EGFR", "KRAS", "TP53", "MDM2"],  # 基因列表
    corr_df=corr_df,                             # 相关性矩阵 DataFrame
    figsize=(10, 8),                            # 图片大小
    fontsize=10,
    cmap="RdBu_r",
    annot=True,                                 # 显示数值
    fmt=".2f",                                   # 数值格式
    linewidths=0.5                              # 线条宽度
)
```

输出：`heatmap_correlation_matrix.png`

## 完整工作流示例

```python
import scanpy as sc
from gene_correlation_ultra import compute_correlation_numba
from gene_correlation_ultra.gene_correlation_plot import CorrelationVisualizer

# 1. 加载数据
adata = sc.read_h5ad("your_data.h5ad")

# 2. 执行相关性分析
target_genes = ["EGFR", "KRAS", "TP53", "MDM2", "BRAF"]
de_genes = ["GENE1", "GENE2", "GENE3", "GENE4", "GENE5"]

corr_df, pval_df, sig_pairs = compute_correlation_numba(
    st_expr_matrix=adata,
    target_genes=target_genes,
    de_genes=de_genes,
    method="spearman",
    threshold_p=0.05,
    output_dir="results",
    verbose=True
)

# 3. 初始化可视化器
visualizer = CorrelationVisualizer(
    st_expr_matrix=adata,
    output_dir="correlation_plots"
)

# 4. 可视化单基因对
visualizer.plot_single_pair_scatter("EGFR", "KRAS")

# 5. 可视化 Top 12 基因对
visualizer.plot_top_pairs_scatter_grid(sig_pairs, top_n=12, n_cols=4)

# 6. 可视化完整矩阵
visualizer.plot_correlation_matrix_heatmap(
    gene_list=target_genes,
    corr_df=corr_df
)

print("可视化完成！结果保存在 correlation_plots/ 目录")
```

## 输出文件说明

| 文件名 | 说明 |
|--------|------|
| `scatter_GENE1_GENE2.png` | 单基因对散点图 |
| `scatter_grid_topN.png` | Top N 基因对网格图 |
| `heatmap_gene_pairs.png` | 多基因对热图 |
| `heatmap_correlation_matrix.png` | 完整相关性矩阵热图 |

## 自定义样式

### 颜色主题

```python
# 设置颜色主题
visualizer = CorrelationVisualizer(
    st_expr_matrix=adata,
    output_dir="correlation_plots",
    cmap="RdBu_r"      # 全局颜色映射
)

# 单独设置热图颜色
visualizer.plot_correlation_matrix_heatmap(
    ...,
    cmap="coolwarm",   # 覆盖全局设置
    vmin=-1,
    vmax=1
)
```

### 字体和样式

```python
# 调整字体大小和图片大小
visualizer.plot_single_pair_scatter(
    "EGFR", "KRAS",
    fontsize=14,           # 字体大小
    figsize=(12, 12),     # 图片大小
    dpi=300              # 分辨率
)
```

### 热图标注

```python
# 显示相关系数数值
visualizer.plot_correlation_matrix_heatmap(
    ...,
    annot=True,      # 显示数值
    fmt=".3f",       # 保留3位小数
    linewidths=0.5,  # 网格线宽度
    linecolor="gray"
)
```

## 性能优化

### 大规模数据

当基因数量很多时（>100），建议：

```python
# 只可视化显著相关对
sig_pairs_subset = sig_pairs.nsmallest(50, 'p_value')

visualizer.plot_top_pairs_scatter_grid(
    sig_pairs_subset,
    top_n=20,
    n_cols=5
)
```

### 降低分辨率

对于预览或快速检查：

```python
visualizer = CorrelationVisualizer(
    st_expr_matrix=adata,
    output_dir="correlation_plots",
    dpi=150        # 降低分辨率
)
```

## 常见问题

### Q: 如何处理缺失值（NaN）？

A: 可视化器会自动跳过包含 NaN 的数据点。

### Q: 如何调整热图的颜色范围？

A: 使用 `vmin` 和 `vmax` 参数：

```python
visualizer.plot_correlation_matrix_heatmap(
    ...,
    vmin=-0.5,  # 只显示 -0.5 到 0.5 的范围
    vmax=0.5
)
```

### Q: 如何保存为 PDF 矢量格式？

A: 修改文件扩展名即可：

```python
# 在函数内部，将 .png 改为 .pdf
# 例如 scatter_EGFR_KRAS.pdf
```

### Q: 如何批量生成所有基因对的散点图？

A: 遍历 `sig_pairs`：

```python
for _, row in sig_pairs.iterrows():
    gene_x = row['gene_x']
    gene_y = row['gene_y']
    visualizer.plot_single_pair_scatter(gene_x, gene_y)
```

## 依赖项

- numpy
- pandas
- matplotlib
- seaborn
- scanpy

## 更多示例

查看 `examples/run_ultra.py` 了解完整工作流示例。
