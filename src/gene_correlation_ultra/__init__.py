"""Gene Correlation Ultra - 极致优化的空间转录组基因相关性分析工具

这个包提供了使用 Numba 加速的基因相关性计算功能，
包括 Pearson 和 Spearman 相关系数计算。
"""

from .gene_correlation_ultra import compute_correlation_numba

__version__ = "1.0.0"
__author__ = "LiangYu"

__all__ = [
    "compute_correlation_numba",
]
