import numpy as np
from scipy.stats import spearmanr
import warnings
warnings.filterwarnings("ignore")

try:
    from numba import njit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    print("Numba 不可用，无法测试")
    exit(1)

# Numba 秩计算 (使用 float64 提高精度)
@njit(fastmath=True, cache=True)
def _get_ranks(values: np.ndarray) -> np.ndarray:
    n = len(values)
    ranks = np.empty(n, dtype=np.float64)
    sorted_indices = np.argsort(values)
    
    i = 0
    while i < n:
        j = i
        while j < n and values[sorted_indices[j]] == values[sorted_indices[i]]:
            j += 1
        
        avg_rank = (i + j - 1) / 2.0 + 1.0
        for k in range(i, j):
            ranks[sorted_indices[k]] = avg_rank
        i = j
    
    return ranks

# Numba Spearman (使用 float64 中间计算)
@njit(fastmath=True, cache=True)
def spearman_numba(x: np.ndarray, y: np.ndarray):
    valid_mask = ~np.isnan(x) & ~np.isnan(y)
    n_valid = valid_mask.sum()
    
    if n_valid < 3:
        return np.float32(np.nan)
    
    # 提取有效数据，使用 float64
    x_valid = np.empty(n_valid, dtype=np.float64)
    y_valid = np.empty(n_valid, dtype=np.float64)
    idx = 0
    for i in range(len(x)):
        if valid_mask[i]:
            x_valid[idx] = x[i]
            y_valid[idx] = y[i]
            idx += 1
    
    # 计算秩
    x_ranks = _get_ranks(x_valid)
    y_ranks = _get_ranks(y_valid)
    
    # 手动计算均值 (避免 .mean() 的精度问题)
    x_sum = 0.0
    y_sum = 0.0
    for i in range(n_valid):
        x_sum += x_ranks[i]
        y_sum += y_ranks[i]
    
    x_mean = x_sum / n_valid
    y_mean = y_sum / n_valid
    
    # 计算协方差和方差
    cov = 0.0
    var_x = 0.0
    var_y = 0.0
    for i in range(n_valid):
        dx = x_ranks[i] - x_mean
        dy = y_ranks[i] - y_mean
        cov += dx * dy
        var_x += dx * dx
        var_y += dy * dy
    
    if var_x == 0 or var_y == 0:
        return np.float32(np.nan)
    
    r = cov / np.sqrt(var_x * var_y)
    return np.float32(r)

print("="*70)
print("Spearman 相关性一致性测试 (修复版)")
print("="*70)

# 测试用例
test_cases = [
    ("简单线性正相关", np.array([1, 2, 3, 4, 5], dtype=np.float32), np.array([2, 4, 6, 8, 10], dtype=np.float32)),
    ("简单线性负相关", np.array([1, 2, 3, 4, 5], dtype=np.float32), np.array([10, 8, 6, 4, 2], dtype=np.float32)),
    ("并列值", np.array([1, 2, 2, 4, 5], dtype=np.float32), np.array([1, 3, 3, 5, 7], dtype=np.float32)),
    ("含NaN", np.array([1, 2, np.nan, 4, 5], dtype=np.float32), np.array([2, 4, 6, 8, 10], dtype=np.float32)),
    ("随机数据", np.array([3, 1, 4, 1, 5, 9, 2, 6], dtype=np.float32), np.array([2, 7, 1, 8, 2, 8, 1, 8], dtype=np.float32)),
    ("小样本", np.array([1.0, 2.0, 3.0], dtype=np.float32), np.array([2.0, 4.0, 6.0], dtype=np.float32)),
]

all_match = True
tolerance = 1e-8

for name, x, y in test_cases:
    # Numba 计算
    r_numba = spearman_numba(x, y)
    
    # SciPy 计算
    mask = ~(np.isnan(x) | np.isnan(y))
    r_scipy, _ = spearmanr(x[mask], y[mask])
    
    diff = abs(r_numba - r_scipy)
    match = diff < tolerance
    all_match = all_match and match
    
    print(f"\n{name}:")
    print(f"  Numba:  {r_numba:.10f}")
    print(f"  SciPy:  {r_scipy:.10f}")
    print(f"  差异:   {diff:.2e}")
    print(f"  状态:   {'✅ 一致' if match else '❌ 不一致'}")

print("\n" + "="*70)
if all_match:
    print("✅ 所有测试通过！Numba 实现与 SciPy 结果一致")
else:
    print("❌ 部分测试失败！需要检查实现")
print("="*70)

