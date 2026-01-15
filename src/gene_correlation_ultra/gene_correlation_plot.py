import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.sparse as sp
from scipy.stats import pearsonr, spearmanr, kendalltau
from typing import Union, Tuple, Optional, List
import logging
import os


def plot_correlation(
    x: Union[list, np.ndarray, pd.Series, int, float],
    y: Union[list, np.ndarray, pd.Series, int, float],
    xlabel: str = 'X',
    ylabel: str = 'Y',
    title: str = 'Correlation Plot',
    method: str = 'pearson',
    trendline_color: str = 'red',
    trendline_style: str = '-',
    trendline_width: float = 2,
    scatter_color: str = 'black',
    scatter_size: float = 30,
    scatter_alpha: float = 1.0,
    ci_alpha: float = 0.2,
    ci_color: Optional[str] = None,
    font_scale: float = 1.0,
    title_fontsize: float = 12,
    label_fontsize: float = 10,
    tick_fontsize: float = 8,
    annot_fontsize: float = 10,
    figure_size: Tuple[float, float] = (3, 3),
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
    show_trendline: bool = True,
    show_ci: bool = True,
    save_path: Optional[str] = None,
    dpi: int = 300,
    tight_layout: bool = True
) -> Tuple[plt.Figure, plt.Axes]:
    """
    绘制带相关性标注的高级散点图（支持Pearson/Spearman/Kendall相关系数）
    核心优化：
    1.  移除标注框，文字无背景
    2.  标注位置随散点分布自动调整
    3.  黑色实心点 + 小画布 + 斜体新罗马字体 + 可自定义字号
    """
    # ====================== 数据预处理与校验 ======================
    def to_1d_array(data):
        arr = np.asarray(data)
        if arr.ndim == 0:
            arr = arr.reshape(1)
        elif arr.ndim > 1:
            raise ValueError(f"输入数据维度错误！仅支持1维序列，当前维度: {arr.ndim}")
        return arr
    
    x_arr = to_1d_array(x)
    y_arr = to_1d_array(y)
    
    if len(x_arr) != len(y_arr):
        raise ValueError(f"X和Y数据长度不一致！X长度={len(x_arr)}, Y长度={len(y_arr)}")
    if len(x_arr) == 0:
        raise ValueError("输入数据不能为空！")
    if not np.issubdtype(x_arr.dtype, np.number) or not np.issubdtype(y_arr.dtype, np.number):
        raise ValueError("X和Y必须为数值型数据！")
    
    valid_methods = ['pearson', 'spearman', 'kendall']
    if method not in valid_methods:
        raise ValueError(f"method必须是{valid_methods}中的一种，当前输入: {method}")
    
    # ====================== 自适应配置 ======================
    valid_mask = ~(np.isnan(x_arr) | np.isnan(y_arr))
    valid_count = np.sum(valid_mask)
    
    if valid_count < 2:
        show_trendline = False
        show_ci = False
        if valid_count == 0:
            raise ValueError("有效数据点为0（全部是NaN）！")
        elif valid_count == 1:
            print("警告：仅1个有效数据点，无法计算相关性和绘制趋势线")
    
    # ====================== 字体配置（新罗马+斜体） ======================
    def get_times_font():
        import matplotlib.font_manager as fm
        font_names = ['Times New Roman', 'Times', 'DejaVu Serif']
        for name in font_names:
            if name in [f.name for f in fm.fontManager.ttflist]:
                return name
        return 'serif'
    times_font = get_times_font()
    
    sns.set_style("white", {"font_scale": font_scale})
    plt.rcParams.update({
        'font.family': times_font,
        'font.size': annot_fontsize,
        'axes.labelsize': label_fontsize,
        'axes.titlesize': title_fontsize,
        'xtick.labelsize': tick_fontsize,
        'ytick.labelsize': tick_fontsize,
        'axes.unicode_minus': False
    })
    
    # ====================== 创建画布 ======================
    fig, ax = plt.subplots(figsize=figure_size)
    
    # ====================== 绘制趋势线和置信区间 ======================
    if show_trendline and valid_count >= 2:
        ci = 95 if show_ci else None
        ci_color = ci_color or trendline_color
        
        sns.regplot(
            x=x_arr,
            y=y_arr,
            ax=ax,
            scatter=False,
            line_kws={
                'color': trendline_color,
                'linestyle': trendline_style,
                'linewidth': trendline_width
            },
            ci=ci,
            color=ci_color,
            scatter_kws={'alpha': 0}
        )
        
        for child in ax.get_children():
            if isinstance(child, plt.Polygon):
                child.set_alpha(ci_alpha)
                child.set_color(ci_color)
    
    # ====================== 绘制黑色实心散点 ======================
    ax.scatter(
        x_arr,
        y_arr,
        color=scatter_color,
        s=scatter_size,
        alpha=scatter_alpha,
        edgecolors='none'
    )
    
    # ====================== 相关性标注（无框+自动调整位置） ======================
    def get_best_annot_pos(ax, x_data, y_data):
        """
        根据散点分布自动选择最佳标注位置（相对坐标）
        候选位置：左上/左下/右上/右下 → 选择离散点最远的位置
        """
        # 候选位置（相对坐标，0-1范围）
        candidate_pos = [
            (0.1, 0.9),  # 左上
            (0.1, 0.1),  # 左下
            (0.9, 0.9),  # 右上
            (0.9, 0.1)   # 右下
        ]
        
        # 将相对坐标转为数据坐标
        def trans_rel_to_data(rel_pos):
            return ax.transAxes.transform(rel_pos)
        
        # 计算每个候选位置到所有散点的平均距离
        data_coords = np.column_stack((x_data, y_data))
        avg_distances = []
        for pos in candidate_pos:
            pos_data = trans_rel_to_data(pos)
            # 计算欧氏距离
            distances = np.sqrt(np.sum((data_coords - pos_data)**2, axis=1))
            avg_distances.append(np.mean(distances))
        
        # 选择平均距离最大的位置（散点最稀疏）
        best_idx = np.argmax(avg_distances)
        return candidate_pos[best_idx]
    
    if valid_count >= 2:
        x_valid = x_arr[valid_mask]
        y_valid = y_arr[valid_mask]
        
        if method == 'pearson':
            corr, p_value = pearsonr(x_valid, y_valid)
        elif method == 'spearman':
            corr, p_value = spearmanr(x_valid, y_valid)
        elif method == 'kendall':
            corr, p_value = kendalltau(x_valid, y_valid)
        
        # 格式化p值
        if p_value < 1e-10:
            p_text = 'P < 1e-10'
        elif p_value < 0.001:
            p_text = 'P < 0.001'
        else:
            p_text = f'P = {p_value:.3f}'
        
        # 自动获取最佳标注位置
        best_pos = get_best_annot_pos(ax, x_valid, y_valid)
        
        # 标注文本（斜体+新罗马，无框）
        annot_text = f'{method.capitalize()} $r$ = {corr:.4f}\n{p_text}'
        ax.text(
            best_pos[0],
            best_pos[1],
            annot_text,
            transform=ax.transAxes,
            fontsize=annot_fontsize,
            fontstyle='italic',
            fontfamily=times_font,
            verticalalignment='center',
            horizontalalignment='center'
            # 移除bbox参数 → 无标注框
        )
    
    # ====================== 样式配置 ======================
    ax.set_xlabel(xlabel, fontfamily=times_font, fontsize=label_fontsize)
    ax.set_ylabel(ylabel, fontfamily=times_font, fontsize=label_fontsize)
    ax.set_title(title, fontfamily=times_font, fontsize=title_fontsize)
    
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.grid(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    if tight_layout:
        plt.tight_layout()
    
    # ====================== 保存图片 ======================
    if save_path is not None:
        fig.savefig(
            save_path,
            dpi=dpi,
            bbox_inches='tight' if tight_layout else None,
            facecolor='white',
            edgecolor='none'
        )
        print(f"图片已保存至: {save_path}")
    
    return fig, ax

# ========== 稀疏矩阵转换函数 ==========
def sparse_to_1d_float(sparse_view: sp.spmatrix) -> np.ndarray:
    """将稀疏矩阵/视图转换为1维浮点型密集数组"""
    if not sp.issparse(sparse_view):
        raise TypeError(f"输入不是稀疏矩阵！当前类型: {type(sparse_view)}")
    dense_arr = sparse_view.toarray()
    arr_1d = dense_arr.ravel()
    arr_float = arr_1d.astype(np.float64)
    arr_float[np.isinf(arr_float)] = np.nan
    return arr_float


# ========== 相关性矩阵可视化函数 ==========
def plot_correlation_heatmap(
    corr_matrix: pd.DataFrame,
    pval_matrix: pd.DataFrame,
    threshold_p: float = 0.05,
    figsize: Tuple[int, int] = (10, 8),
    cmap: str = "RdBu_r",
    title: str = "Target Genes vs DE Genes Correlation Heatmap",
    save_path: Optional[str] = None,
    dpi: int = 300
) -> plt.Figure:
    """
    绘制相关性矩阵热图，标注显著相关的单元格
    """
    # 创建显著性掩码（P < threshold_p 标记为*）
    sig_mask = pval_matrix < threshold_p
    # 构建标注文本（相关系数 + 显著性标记）
    annot_text = np.empty_like(corr_matrix.values, dtype=object)
    for i in range(corr_matrix.shape[0]):
        for j in range(corr_matrix.shape[1]):
            corr = f"{corr_matrix.iloc[i, j]:.2f}"
            sig = "*" if sig_mask.iloc[i, j] else ""
            annot_text[i, j] = f"{corr}{sig}"

    # 绘图
    plt.figure(figsize=figsize)
    sns.heatmap(
        corr_matrix,
        annot=annot_text,
        fmt="",  # 不使用自动格式化，直接显示annot_text
        cmap=cmap,
        vmin=-1,
        vmax=1,
        center=0,
        linewidths=0.5,
        cbar_kws={"label": "Correlation Coefficient"},
        square=True
    )
    plt.title(title, fontsize=14, pad=20)
    plt.xlabel("Differentially Expressed Genes", fontsize=12)
    plt.ylabel("Target Genes", fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    # 添加显著性说明
    plt.text(
        0.02, 0.02, 
        "* P < 0.05", 
        transform=plt.gca().transAxes,
        fontsize=10,
        bbox=dict(facecolor="white", alpha=0.8)
    )

    if save_path is not None:
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
        print(f"热图已保存至: {save_path}")
    
    return plt.gcf()


# ========== 基因相关性可视化类 ==========
class CorrelationVisualizer:
    """
    基因相关性结果可视化工具
    
    功能:
    - 单个基因对散点图
    - 多个基因对热图
    - 相关性矩阵热图
    - Top基因对散点图网格
    """
    
    def __init__(
        self,
        st_expr_matrix: Union[pd.DataFrame, np.ndarray],
        output_dir: str = "correlation_plots",
        logger: Optional[logging.Logger] = None
    ):
        """
        参数:
            st_expr_matrix: 空间转录组表达矩阵 (cells × genes)
            output_dir: 图片输出目录
            logger: 日志记录器
        """
        self.st_expr_matrix = st_expr_matrix
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        if logger is None:
            logger = logging.getLogger(__name__)
            logger.setLevel(logging.INFO)
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter('%(message)s'))
            logger.addHandler(handler)
        self.logger = logger
        
        # 转换为DataFrame
        if hasattr(st_expr_matrix, 'obs_names') and hasattr(st_expr_matrix, 'var_names'):
            # AnnData对象
            self.logger.info("[数据准备] 检测到AnnData对象")
            if sp.issparse(st_expr_matrix.X):
                expr = st_expr_matrix.X.toarray()
            else:
                expr = st_expr_matrix.X
            # 转置: genes×spots
            self.expr_df = pd.DataFrame(
                expr.T,
                index=[str(g).upper() for g in st_expr_matrix.var_names],
                columns=st_expr_matrix.obs_names
            )
        elif isinstance(st_expr_matrix, np.ndarray):
            self.expr_df = pd.DataFrame(st_expr_matrix)
        else:
            self.expr_df = st_expr_matrix.copy()
    
    def _get_gene_expression(self, gene_name: str) -> np.ndarray:
        """获取指定基因的表达值"""
        if gene_name in self.expr_df.columns:
            expr = self.expr_df[gene_name].values
        elif gene_name in self.expr_df.index:
            expr = self.expr_df.loc[gene_name].values
        else:
            raise ValueError(f"基因 '{gene_name}' 不在表达矩阵中")
        
        # 处理稀疏矩阵
        if sp.issparse(expr):
            expr = sparse_to_1d_float(expr)
        
        return expr
    
    def plot_single_pair_scatter(
        self,
        gene1: str,
        gene2: str,
        method: str = 'pearson',
        sample_size: Optional[int] = None,
        show_plot: bool = False,
        dpi: int = 300
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        绘制单个基因对的相关性散点图
        
        参数:
            gene1: 基因1名称
            gene2: 基因2名称
            method: 相关系数类型 ('pearson', 'spearman', 'kendall')
            sample_size: 采样数量 (大数据时使用)
            show_plot: 是否显示图片
            dpi: 图片分辨率
        
        返回:
            fig, ax: matplotlib图形对象
        """
        self.logger.info(f"[散点图] 绘制基因对: {gene1} vs {gene2}")
        
        # 获取表达值
        expr1 = self._get_gene_expression(gene1)
        expr2 = self._get_gene_expression(gene2)
        
        # 采样 (大数据)
        if sample_size is not None and len(expr1) > sample_size:
            idx = np.random.choice(len(expr1), sample_size, replace=False)
            expr1 = expr1[idx]
            expr2 = expr2[idx]
            self.logger.info(f"[散点图] 采样: {sample_size} / {len(expr1)}")
        
        # 绘制散点图
        save_path = os.path.join(self.output_dir, f"scatter_{gene1}_{gene2}.png")
        fig, ax = plot_correlation(
            x=expr1,
            y=expr2,
            xlabel=gene1,
            ylabel=gene2,
            title=f"{gene1} vs {gene2}",
            method=method,
            save_path=save_path,
            dpi=dpi
        )
        
        if not show_plot:
            plt.close(fig)
        
        return fig, ax
    
    def plot_top_pairs_scatter_grid(
        self,
        sig_pairs_df: pd.DataFrame,
        top_n: int = 9,
        n_cols: int = 3,
        method: str = 'pearson',
        sample_size: Optional[int] = None,
        dpi: int = 300
    ) -> plt.Figure:
        """
        绘制Top基因对的散点图网格
        
        参数:
            sig_pairs_df: 显著相关对DataFrame (需包含 gene1, gene2 列)
            top_n: Top N基因对
            n_cols: 每列子图数量
            method: 相关系数类型
            sample_size: 采样数量
            dpi: 图片分辨率
        
        返回:
            fig: matplotlib图形对象
        """
        self.logger.info(f"[网格图] 绘制Top {top_n}基因对")
        
        # 选择Top N
        top_pairs = sig_pairs_df.head(top_n)
        
        # 计算行列数
        n_rows = (top_n + n_cols - 1) // n_cols
        
        # 创建网格
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*4, n_rows*4))
        if n_rows == 1 and n_cols == 1:
            axes = np.array([axes])
        axes = axes.flatten()
        
        # 绘制每个基因对
        for idx, (_, row) in enumerate(top_pairs.iterrows()):
            gene1, gene2 = row['gene1'], row['gene2']
            ax = axes[idx]
            
            # 获取表达值
            expr1 = self._get_gene_expression(gene1)
            expr2 = self._get_gene_expression(gene2)
            
            # 采样
            if sample_size is not None and len(expr1) > sample_size:
                idx_rand = np.random.choice(len(expr1), sample_size, replace=False)
                expr1 = expr1[idx_rand]
                expr2 = expr2[idx_rand]
            
            # 计算相关系数
            valid_mask = ~(np.isnan(expr1) | np.isnan(expr2))
            if method == 'pearson':
                corr, pval = pearsonr(expr1[valid_mask], expr2[valid_mask])
            elif method == 'spearman':
                corr, pval = spearmanr(expr1[valid_mask], expr2[valid_mask])
            else:
                corr, pval = kendalltau(expr1[valid_mask], expr2[valid_mask])
            
            # 散点图
            ax.scatter(expr1, expr2, s=10, alpha=0.5, color='black', edgecolors='none')
            
            # 趋势线
            from sklearn.linear_model import LinearRegression
            valid_idx = np.where(valid_mask)[0]
            if len(valid_idx) > 1:
                lr = LinearRegression()
                lr.fit(expr1[valid_idx].reshape(-1, 1), expr2[valid_idx])
                x_line = np.linspace(expr1.min(), expr1.max(), 100)
                y_line = lr.predict(x_line.reshape(-1, 1))
                ax.plot(x_line, y_line, color='red', linewidth=1.5)
            
            # 标注
            p_text = 'P<0.001' if pval < 0.001 else f'P={pval:.3f}'
            ax.text(0.05, 0.95, f'{method.capitalize()} r={corr:.2f}\n{p_text}',
                   transform=ax.transAxes, fontsize=9, verticalalignment='top')
            ax.set_xlabel(gene1, fontsize=10)
            ax.set_ylabel(gene2, fontsize=10)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
        
        # 隐藏多余子图
        for idx in range(top_n, len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        
        # 保存
        save_path = os.path.join(self.output_dir, f"scatter_grid_top{top_n}.png")
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
        self.logger.info(f"[网格图] 保存: {save_path}")
        plt.close(fig)
        
        return fig
    
    def plot_multiple_pairs_heatmap(
        self,
        gene_pairs: List[Tuple[str, str]],
        sig_pairs_df: Optional[pd.DataFrame] = None,
        corr_df: Optional[pd.DataFrame] = None,
        method: str = 'pearson',
        figsize: Tuple[int, int] = (10, 8),
        dpi: int = 300
    ) -> plt.Figure:
        """
        绘制多个基因对的相关性热图
        
        参数:
            gene_pairs: 基因对列表 [(gene1, gene2), ...]
            sig_pairs_df: 显著相关对DataFrame (用于获取相关系数)
            corr_df: 相关性矩阵DataFrame (用于获取相关系数)
            method: 相关系数类型 (未提供相关系数时计算)
            figsize: 图片大小
            dpi: 图片分辨率
        
        返回:
            fig: matplotlib图形对象
        """
        self.logger.info(f"[热图] 绘制 {len(gene_pairs)} 个基因对")
        
        # 获取或计算相关系数
        corrs = []
        for gene1, gene2 in gene_pairs:
            # 优先使用已有相关系数
            if sig_pairs_df is not None:
                mask = (sig_pairs_df['gene1'] == gene1) & (sig_pairs_df['gene2'] == gene2)
                if mask.any():
                    corr = sig_pairs_df.loc[mask, 'correlation'].values[0]
                    corrs.append(corr)
                    continue
            
            if corr_df is not None:
                if gene1 in corr_df.index and gene2 in corr_df.columns:
                    corr = corr_df.loc[gene1, gene2]
                    corrs.append(corr)
                    continue
                elif gene2 in corr_df.index and gene1 in corr_df.columns:
                    corr = corr_df.loc[gene2, gene1]
                    corrs.append(corr)
                    continue
            
            # 实时计算
            expr1 = self._get_gene_expression(gene1)
            expr2 = self._get_gene_expression(gene2)
            valid_mask = ~(np.isnan(expr1) | np.isnan(expr2))
            
            if method == 'pearson':
                corr, _ = pearsonr(expr1[valid_mask], expr2[valid_mask])
            elif method == 'spearman':
                corr, _ = spearmanr(expr1[valid_mask], expr2[valid_mask])
            else:
                corr, _ = kendalltau(expr1[valid_mask], expr2[valid_mask])
            
            corrs.append(corr)
        
        # 构建矩阵
        labels = [f"{g1}\nvs\n{g2}" for g1, g2 in gene_pairs]
        corr_matrix = pd.DataFrame([corrs], columns=labels)
        
        # 绘制热图
        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(
            corr_matrix,
            annot=True,
            fmt='.2f',
            cmap='RdBu_r',
            vmin=-1,
            vmax=1,
            center=0,
            ax=ax,
            cbar_kws={'label': 'Correlation'}
        )
        ax.set_title('Gene Pairs Correlation Heatmap', fontsize=14)
        ax.set_ylabel('Gene Pairs')
        plt.xticks(rotation=0)
        plt.tight_layout()
        
        # 保存
        save_path = os.path.join(self.output_dir, "heatmap_gene_pairs.png")
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
        self.logger.info(f"[热图] 保存: {save_path}")
        plt.close(fig)
        
        return fig
    
    def plot_correlation_matrix_heatmap(
        self,
        genes: List[str],
        corr_df: Optional[pd.DataFrame] = None,
        method: str = 'pearson',
        figsize: Tuple[int, int] = (10, 8),
        dpi: int = 300
    ) -> plt.Figure:
        """
        绘制基因相关性矩阵热图
        
        参数:
            genes: 基因列表
            corr_df: 相关性矩阵DataFrame (如未提供则实时计算)
            method: 相关系数类型 (未提供相关矩阵时计算)
            figsize: 图片大小
            dpi: 图片分辨率
        
        返回:
            fig: matplotlib图形对象
        """
        self.logger.info(f"[矩阵热图] 绘制 {len(genes)} 个基因")
        
        # 获取或计算相关性矩阵
        if corr_df is not None:
            # 从完整矩阵提取子集
            available_genes = [g for g in genes if g in corr_df.index and g in corr_df.columns]
            if len(available_genes) < 2:
                raise ValueError(f"可用基因不足: {len(available_genes)}/{len(genes)}")
            corr_matrix = corr_df.loc[available_genes, available_genes]
        else:
            # 实时计算
            expr_matrix = np.column_stack([self._get_gene_expression(g) for g in genes])
            corr_matrix = pd.DataFrame(
                np.corrcoef(expr_matrix.T),
                index=genes,
                columns=genes
            )
        
        # 绘制热图
        fig = plot_correlation_heatmap(
            corr_matrix=corr_matrix,
            pval_matrix=pd.DataFrame(np.ones_like(corr_matrix), index=corr_matrix.index, columns=corr_matrix.columns),
            figsize=figsize,
            title='Gene Correlation Matrix',
            save_path=os.path.join(self.output_dir, "heatmap_correlation_matrix.png"),
            dpi=dpi
        )
        plt.close(fig)
        
        return fig
