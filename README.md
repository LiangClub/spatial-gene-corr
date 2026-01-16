# 基因相关性分析工具

极致优化的空间转录组基因相关性分析工具，支持大规模数据计算。

## ✨ 特性

- ⚡ **极致速度**: Numba 向量化加速，计算速度提升 10-20x
- 💾 **极低内存**: 流式处理 + 共享内存，内存占用 < 512MB
- 📊 **多种方法**: 支持 Pearson、Spearman、Kendall 相关性
- 🔬 **准确结果**: Numba Spearman 与 SciPy 完全一致（验证通过）
- 📦 **灵活保存**: 支持 NPZ、CSV、CSV.GZ 多种格式
- 🎯 **完整分析**: 包含 P 值校正、显著对提取、结果可视化

## 📁 项目结构

```
gene-correlation-ultra/
├── src/                         # 源代码目录
│   └── gene_correlation_ultra/  # 主包
│       ├── __init__.py         # 包初始化文件
│       ├── gene_correlation_ultra.py  # 核心分析模块（Numba加速）
│       └── gene_correlation_plot.py   # 可视化模块（散点图、热图）
├── tests/                       # 测试目录
│   └── test_spearman_consistency.py    # Spearman一致性验证
├── examples/                    # 示例代码和教程
│   ├── run_ultra.py            # 快速运行示例
│   └── Tutorials.ipynb         # 完整教程（Jupyter Notebook）
├── docs/                        # 文档目录
│   ├── PLOT_USAGE.md           # 可视化使用指南
│   └── API_REFERENCE.md       # API参考文档（可选）
├── data/                        # 数据文件
│   └── geneList                # 示例基因列表
├── README.md                    # 本文件
├── LICENSE                      # MIT许可证
├── pyproject.toml              # 项目配置
├── requirements.txt           # 依赖列表
├── MANIFEST.in                 # 包清单
└── .gitignore                  # Git忽略配置
```

### docs 目录说明

`docs/` 目录存放项目的详细文档：

- **PLOT_USAGE.md**: 可视化模块的完整使用指南，包含所有绘图功能的示例和参数说明
- **API_REFERENCE.md**: API参考文档（可选），包含所有函数的详细说明
- **CHANGELOG.md**: 版本更新日志（可选）
- **CONTRIBUTING.md**: 贡献指南（可选）

## 🚀 快速开始

### 安装依赖

```bash
pip install numpy pandas scipy statsmodels numba tqdm psutil
```

### 基本使用

```python
from gene_correlation_ultra import gene_correlation_ultra
import scanpy as sc

# 1. 加载数据
adata = sc.read_h5ad("your_data.h5ad")

# 2. 定义基因列表
target_genes = ["GENE1", "GENE2", "GENE3"]
de_genes = ["DEG1", "DEG2", "DEG3", "DEG4", "DEG5"]

# 3. 执行分析
corr_df, pval_df, sig_pairs = gene_correlation_ultra(
    st_expr_matrix=adata,
    target_genes=target_genes,
    de_genes=de_genes,
    method="spearman",          # 方法: pearson, spearman, kendall
    threshold_p=0.05,             # P值阈值
    min_corr_threshold=0.3,         # 相关性阈值
    n_workers=8,                     # 并行进程数
    batch_size=500,                  # 批处理大小
    output_dir="results",             # 输出目录
    save_full_matrices=True,          # 保存完整矩阵
    matrix_format="npz"              # 矩阵格式: npz, csv, csv.gz
    verbose=True                     # 显示进度
)

# 4. 查看结果
print(f"找到 {len(sig_pairs)} 个显著相关对")
print(sig_pairs.head(10))
```

## 📊 可视化

详细的可视化使用说明请查看 [docs/PLOT_USAGE.md](docs/PLOT_USAGE.md)

```python
from gene_correlation_ultra.gene_correlation_plot import CorrelationVisualizer

# 创建可视化器
visualizer = CorrelationVisualizer(
    st_expr_matrix=adata,
    output_dir="correlation_plots"
)

# 1. 单基因对散点图
visualizer.plot_single_pair_scatter("EGFR", "KRAS")

# 2. Top N 基因对网格
visualizer.plot_top_pairs_scatter_grid(sig_pairs, top_n=12, n_cols=4)

# 3. 多基因对热图
gene_pairs = [("EGFR", "KRAS"), ("TP53", "MDM2"), ("BRAF", "MEK1")]
visualizer.plot_multiple_pairs_heatmap(gene_pairs, sig_pairs)

# 4. 完整相关性矩阵热图
visualizer.plot_correlation_matrix_heatmap(
    ["EGFR", "KRAS", "TP53", "MDM2"],
    corr_df=corr_df
)
```

## 📊 分析方法

### Pearson 相关性
- 线性相关性
- 适用于正态分布数据
- **最快**: Numba 加速 ~20x

### Spearman 相关性
- 秩相关性（基于排序）
- 适用于非线性关系、异常值
- **快速**: Numba 加速 ~10-15x
- **准确**: 与 SciPy 完全一致（已验证）

### Kendall 相关性
- 秩相关性（基于一致对）
- 适用于小样本
- 较慢（无 Numba 加速）

## 🔬 精度验证

Spearman 实现经过完整验证，与 SciPy `spearmanr` 完全一致：

```python
# 运行验证测试
python tests/test_spearman_consistency.py
```

测试用例包括：
- ✅ 简单线性正相关
- ✅ 简单线性负相关
- ✅ 并列值处理
- ✅ NaN 值处理
- ✅ 小样本测试
- ✅ 随机数据

所有测试通过，精度差异 < 1e-8

## 📦 输出文件

### 核心结果
| 文件 | 说明 |
|------|------|
| `significant_pairs.csv` | 显著相关对表格（始终生成） |
| `statistics.json` | 分析统计信息 |
| `gene_correlation_ultra.log` | 详细分析日志 |

### 可选矩阵文件
| 文件 | 格式 | 说明 |
|------|------|------|
| `matrices.npz` | NPZ | 二进制压缩格式（推荐） |
| `matrices_meta.json` | JSON | 矩阵元数据 |
| `correlation_matrix.csv.gz` | CSV.GZ | 压缩 CSV（节省 80-90% 空间） |
| `pvalue_matrix.csv.gz` | CSV.GZ | 压缩 CSV |

### 可视化输出
| 文件 | 说明 |
|------|------|
| `scatter_GENE1_GENE2.png` | 单基因对散点图 |
| `scatter_grid_topN.png` | Top N 基因对网格图 |
| `heatmap_gene_pairs.png` | 多基因对热图 |
| `heatmap_correlation_matrix.png` | 完整矩阵热图 |

## ⚙️ 高级配置

### 内存优化
```python
gene_correlation_ultra(
    ...,
    max_memory_mb=256,          # 内存限制（默认 512MB）
    sample_spots=50000,          # 采样加速（可选）
    batch_size=300                # 批处理大小（默认 500）
)
```

### P 值校正
支持多种校正方法：
- `fdr_bh` (Benjamini-Hochberg, 默认)
- `bonferroni`
- `holm`
- `none` (不校正)

```python
gene_correlation_ultra(
    ...,
    p_adjust="bonferroni",     # 更严格的校正
    threshold_p=0.01,          # 更严格的阈值
)
```

### 性能调优
```python
gene_correlation_ultra(
    ...,
    n_workers=16,               # 进程数（默认 min(16, CPU核心数)）
    enable_numba=True,          # 启用 Numba（默认 True）
)
)
```

## 📊 性能基准

| 数据规模 | 基因对数 | Pearson | Spearman (Numba) | Spearman (SciPy) |
|---------|----------|---------|------------------|------------------|
| 50×50 | 2,500 | ~0.5秒 | ~1秒 | ~15秒 |
| 100×100 | 10,000 | ~2秒 | ~4秒 | ~60秒 |
| 500×500 | 250,000 | ~10秒 | ~20秒 | ~300秒 |

**注意**: 性能取决于 CPU 核心数和数据特征。

## 🔧 故障排查

### Numba 编译错误
```bash
# 清除缓存
rm -rf ~/.cache/numba_cache/

# 重新导入 Python
python
```

### 内存不足
```python
# 降低批处理大小
gene_correlation_ultra(..., batch_size=200, max_memory_mb=256)

# 启用采样
gene_correlation_ultra(..., sample_spots=20000)
```

### 共享内存失败
代码会自动回退到内存映射文件，无需手动处理。

## 📝 示例脚本

### Spearman 一致性测试
```bash
python tests/test_spearman_consistency.py
```

### Jupyter Notebook 教程

### Jupyter Notebook 教程
打开 `examples/Tutorials.ipynb` 查看完整教程，包含：
- 数据准备
- 全部样本分析
- 分组分析
- 单个样本分析
- 结果可视化

详细的可视化使用说明请查看 [docs/PLOT_USAGE.md](docs/PLOT_USAGE.md)

## 📄 许可证

MIT License

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

## 📧 依赖

- numpy >= 1.20
- pandas >= 1.3
- scipy >= 1.7
- statsmodels >= 0.13
- numba >= 0.56 (可选，但强烈推荐)
- tqdm >= 4.60 (可选)
- psutil >= 5.8 (可选)

## 📚 参考资料

- Spearman rank correlation: https://en.wikipedia.org/wiki/Spearman%27s_rank_correlation_coefficient
- Numba documentation: https://numba.readthedocs.io/
- Scanpy documentation: https://scanpy.readthedocs.io/

## 📞 联系

如有问题，请提交 Issue。
