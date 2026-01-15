import numpy as np
import pandas as pd
import scanpy as sc
import sys
import os


from gene_correlation_ultra import gene_correlation_ultra, GeneCorrelationUltra


adata = sc.read_h5ad("../subcluster/Malignan.NMF.allsample.h5ad")
target_gene = pd.read_csv("./geneList",sep="\t",index_col=None)
deg = pd.read_csv("../../../05.group2DEGs/DEG/de_wilcoxon.group.Responder-vs-Non-responder.sig_filtered.csv")

target_genes = target_gene["Gene"].to_numpy()
de_genes = deg["geneName"].to_numpy()

# 创建分析器
analyzer = GeneCorrelationUltra(
    method="pearson",
    p_adjust="fdr_bh",
    threshold_p=0.05,
    batch_size=5000,
    use_dtype="float32",
    n_workers=16,
    enable_numba=True,
    min_corr_threshold = 0,
    max_memory_mb = 50720,
    verbose=True
)


# 执行计算
corr_df, pval_df, sig_pairs = analyzer.compute(
    st_expr_matrix=adata,
    target_genes=target_genes,
    de_genes=de_genes
)


# 保存结果
analyzer.save_results(
    corr_df=corr_df,
    pval_df=pval_df,
    sig_pairs=sig_pairs,
    stats= analyzer.stats,
    output_dir="results",
    save_full_matrices = True,
    matrix_format = "csv"
)