# API 参考文档

本文档提供 `gene_correlation_ultra` 包的详细 API 参考。

## 核心函数

### compute_correlation_numba

计算基因相关性矩阵（Numba 加速）。

```python
def compute_correlation_numba(
    st_expr_matrix: AnnData,
    target_genes: List[str],
    de_genes: List[str],
    method: str = "pearson",
    threshold_p: float = 0.05,
    min_corr_threshold: float = 0.0,
    n_workers: int = None,
    batch_size: int = 500,
    max_memory_mb: int = 512,
    sample_spots: int = None,
    output_dir: str = "results",
    save_full_matrices: bool = False,
    matrix_format: str = "npz",
    p_adjust: str = "fdr_bh",
    verbose: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
```

**参数**：

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `st_expr_matrix` | AnnData | 必需 | 空间转录组数据对象 |
| `target_genes` | List[str] | 必需 | 目标基因列表 |
| `de_genes` | List[str] | 必需 | 差异表达基因列表 |
| `method` | str | "pearson" | 相关性方法: "pearson", "spearman", "kendall" |
| `threshold_p` | float | 0.05 | P值显著性阈值 |
| `min_corr_threshold` | float | 0.0 | 最小相关性阈值 |
| `n_workers` | int | None | 并行进程数，默认 min(16, CPU核心数) |
| `batch_size` | int | 500 | 批处理大小 |
| `max_memory_mb` | int | 512 | 最大内存限制（MB） |
| `sample_spots` | int | None | 采样spot数量（加速用） |
| `output_dir` | str | "results" | 输出目录 |
| `save_full_matrices` | bool | False | 是否保存完整矩阵 |
| `matrix_format` | str | "npz" | 矩阵格式: "npz", "csv", "csv.gz" |
| `p_adjust` | str | "fdr_bh" | P值校正方法: "fdr_bh", "bonferroni", "holm", "none" |
| `verbose` | bool | True | 是否显示进度信息 |

**返回值**：

- `corr_df`: 相关性矩阵 DataFrame (n_target_genes × n_de_genes)
- `pval_df`: P值矩阵 DataFrame (n_target_genes × n_de_genes)
- `sig_pairs`: 显著相关对 DataFrame

**sig_pairs 列说明**：

| 列名 | 说明 |
|------|------|
| `gene_x` | X轴基因 |
| `gene_y` | Y轴基因 |
| `correlation` | 相关系数 |
| `p_value` | 原始P值 |
| `p_adjusted` | 校正后P值 |
| `significant` | 是否显著（布尔值） |

## 可视化类

### CorrelationVisualizer

相关性分析结果可视化类。

```python
class CorrelationVisualizer:
    def __init__(
        self,
        st_expr_matrix: AnnData,
        output_dir: str = "correlation_plots",
        dpi: int = 300,
        figsize: Tuple[int, int] = (8, 8)
    )
```

**参数**：

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `st_expr_matrix` | AnnData | 必需 | 空间转录组数据对象 |
| `output_dir` | str | "correlation_plots" | 输出目录 |
| `dpi` | int | 300 | 图片分辨率 |
| `figsize` | Tuple[int, int] | (8, 8) | 默认图片大小 |

### plot_single_pair_scatter

绘制单个基因对的散点图。

```python
def plot_single_pair_scatter(
    self,
    gene_x: str,
    gene_y: str,
    corr: float = None,
    pvalue: float = None,
    figsize: Tuple[int, int] = None,
    fontsize: int = 12,
    show_plot: bool = True
) -> None
```

**参数**：

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `gene_x` | str | 必需 | X轴基因名称 |
| `gene_y` | str | 必需 | Y轴基因名称 |
| `corr` | float | None | 相关系数（可选） |
| `pvalue` | float | None | P值（可选） |
| `figsize` | Tuple[int, int] | None | 图片大小 |
| `fontsize` | int | 12 | 字体大小 |
| `show_plot` | bool | True | 是否显示图片 |

### plot_top_pairs_scatter_grid

绘制Top N基因对的网格散点图。

```python
def plot_top_pairs_scatter_grid(
    self,
    sig_pairs: pd.DataFrame,
    top_n: int = 12,
    n_cols: int = 4,
    figsize: Tuple[int, int] = None,
    fontsize: int = 10
) -> None
```

**参数**：

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `sig_pairs` | pd.DataFrame | 必需 | 显著相关对数据（需包含gene_x, gene_y列） |
| `top_n` | int | 12 | 展示前N个基因对 |
| `n_cols` | int | 4 | 每列图片数 |
| `figsize` | Tuple[int, int] | None | 总图片大小 |
| `fontsize` | int | 10 | 字体大小 |

### plot_multiple_pairs_heatmap

绘制多基因对的热图。

```python
def plot_multiple_pairs_heatmap(
    self,
    gene_pairs: List[Tuple[str, str]],
    sig_pairs: pd.DataFrame,
    figsize: Tuple[int, int] = (12, 10),
    fontsize: int = 12,
    cmap: str = "RdBu_r",
    vmin: float = -1,
    vmax: float = 1
) -> None
```

**参数**：

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `gene_pairs` | List[Tuple[str, str]] | 必需 | 基因对列表 |
| `sig_pairs` | pd.DataFrame | 必需 | 显著相关对数据 |
| `figsize` | Tuple[int, int] | (12, 10) | 图片大小 |
| `fontsize` | int | 12 | 字体大小 |
| `cmap` | str | "RdBu_r" | 颜色映射 |
| `vmin` | float | -1 | 最小值 |
| `vmax` | float | 1 | 最大值 |

### plot_correlation_matrix_heatmap

绘制完整相关性矩阵热图。

```python
def plot_correlation_matrix_heatmap(
    self,
    gene_list: List[str],
    corr_df: pd.DataFrame,
    figsize: Tuple[int, int] = (10, 8),
    fontsize: int = 10,
    cmap: str = "RdBu_r",
    annot: bool = True,
    fmt: str = ".2f",
    linewidths: float = 0.5,
    linecolor: str = "gray"
) -> None
```

**参数**：

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `gene_list` | List[str] | 必需 | 基因列表 |
| `corr_df` | pd.DataFrame | 必需 | 相关性矩阵 |
| `figsize` | Tuple[int, int] | (10, 8) | 图片大小 |
| `fontsize` | int | 10 | 字体大小 |
| `cmap` | str | "RdBu_r" | 颜色映射 |
| `annot` | bool | True | 是否显示数值 |
| `fmt` | str | ".2f" | 数值格式 |
| `linewidths` | float | 0.5 | 线条宽度 |
| `linecolor` | str | "gray" | 线条颜色 |

## 数据格式说明

### 输入数据要求

`st_expr_matrix` 应为 Scanpy AnnData 对象：

- 必须包含 `.X` 属性（表达矩阵）
- 基因名称在 `.var_names` 中
- 空间位点在 `.obs_names` 中

### 输出文件格式

#### significant_pairs.csv

```csv
gene_x,gene_y,correlation,p_value,p_adjusted,significant
EGFR,KRAS,0.85,0.0001,0.0005,True
TP53,MDM2,0.72,0.0003,0.0015,True
```

#### matrices.npz

NumPy 压缩格式，包含：
- `correlation`: 相关性矩阵
- `pvalue`: P值矩阵

加载方式：
```python
import numpy as np
data = np.load("results/matrices.npz")
corr_matrix = data["correlation"]
pval_matrix = data["pvalue"]
```

#### correlation_matrix.csv / pvalue_matrix.csv

完整的CSV格式矩阵，基因名作为行和列索引。

## 性能优化建议

### 大规模数据（>1000基因）

```python
compute_correlation_numba(
    ...,
    batch_size=300,      # 降低批处理大小
    sample_spots=20000,  # 启用采样
    max_memory_mb=256    # 降低内存限制
)
```

### 多核优化

```python
compute_correlation_numba(
    ...,
    n_workers=16  # 使用更多CPU核心
)
```

### 可视化优化

```python
visualizer = CorrelationVisualizer(
    ...,
    dpi=150  # 降低分辨率加速
)
```

## 异常处理

### 常见错误

1. **MemoryError**: 内存不足
   - 解决：减小 `batch_size` 或启用 `sample_spots`

2. **ValueError**: 基因不在数据中
   - 检查基因列表是否正确

3. **RuntimeError**: 共享内存失败
   - 代码会自动回退到内存映射文件

## 版本信息

- **版本**: 1.0.0
- **最低Python版本**: 3.8
- **最低Numba版本**: 0.56.0

## 更多信息

- [可视化使用指南](PLOT_USAGE.md)
- [项目主页](../README.md)
