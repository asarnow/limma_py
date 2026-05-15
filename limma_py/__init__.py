"""
limma_py - Python implementation of R's limma package for differential expression analysis

A comprehensive Python port of the popular R limma package for analyzing gene expression data,
particularly microarray and RNA-seq data. Provides tools for linear modeling, empirical Bayes
moderation, and differential expression testing.
"""

__version__ = "0.1.0"
__author__ = "Zhang Xian"
__email__ = "2967569628@qq.com"

# Import core functionality
from .lmfit import lmFit
from .ebayes import eBayes
from .toptable import toptable
from .test_functions import classify_tests_f
from .contrasts import contrasts_fit
from .makeContrasts import make_contrasts

# Define the public API
__all__ = [
    "lmFit",
    "eBayes",
    "toptable",
    "classify_tests_f",
    "contrasts_fit",
    "make_contrasts"
]

# Package-level docstring
__doc__ = """
limma_py: Python implementation of the R limma package

This package provides a complete Python implementation of the widely-used R limma
(Linear Models for Microarray Data) package for the analysis of gene expression data.

Main Features:
-------------
- Linear model fitting for gene expression data
- Empirical Bayes moderation of standard errors
- Differential expression analysis with multiple testing correction
- Contrast analysis for complex experimental designs
- Comprehensive result reporting with toptable outputs

Core Functions:
--------------
lmFit(object, design=None, **kwargs)
    Fit linear models for each gene in expression data
    
eBayes(fit, proportion=0.01, trend=False, robust=False, **kwargs)
    Empirical Bayes moderation of the standard errors
    
toptable(fit, coef=1, number=10, **kwargs)
    Extract top-ranked genes from linear model fit
    
contrasts_fit(fit, contrasts=None, coefficients=None)
    Compute contrasts for linear model fits
    
make_contrasts(*args, contrasts=None, levels)
    Construct contrast matrices for specified comparisons

Typical Workflow:
----------------
1. Fit linear model: fit = lmFit(expression_data, design_matrix)
2. Compute contrasts: fit = contrasts_fit(fit, contrast_matrix)  
3. Apply empirical Bayes: eb_fit = eBayes(fit)
4. Extract results: results = toptable(eb_fit)

Example Usage:
-------------
>>> import limma_py as limma
>>> import pandas as pd
>>> import numpy as np

# Load expression data and design matrix
>>> data = pd.read_csv("expression_data.csv", header=0)
>>> gene_names = data.iloc[:, 0].values
>>> expr_data = data.iloc[:, 1:]
>>> group = np.array(["Control"] * 3 + ["Treatment"] * 3)

# Create design matrix
>>> design_df = pd.get_dummies(group, drop_first=False)
>>> design = design_df.astype(int)
>>> expr_matrix = expr_data.copy()
>>> expr_matrix.index = gene_names

# Perform differential expression analysis
>>> fit = limma.lmFit(expr_matrix, design)
>>> contrasts = limma.make_contrasts('Treatment - Control', levels=design)
>>> fit = limma.contrasts_fit(fit, contrasts)
>>> eb_fit = limma.eBayes(fit)
>>> results = limma.toptable(eb_fit, number=20)

# Access top differentially expressed genes
>>> print(results.head())

Input/Output Formats:
-------------------
- Expression data: pandas DataFrame (genes × samples) or numpy array
- Design matrix: pandas DataFrame or numpy array  
- Results: pandas DataFrame with comprehensive statistics

Statistical Outputs:
------------------
- logFC: log2 fold change
- AveExpr: average expression level
- t: moderated t-statistic
- P.Value: p-value
- adj.P.Val: adjusted p-value (FDR)
- B: log-odds of differential expression

References:
----------
- Smyth, G. K. (2004). Linear models and empirical bayes methods for assessing 
  differential expression in microarray experiments. Statistical Applications 
  in Genetics and Molecular Biology, 3(1), Article 3.
- Ritchie, M. E., et al. (2015). limma powers differential expression analyses 
  for RNA-sequencing and microarray studies. Nucleic Acids Research, 43(7), e47.

See Also:
--------
For detailed function documentation, use help() on individual functions:
>>> help(limma.lmFit)
>>> help(limma.eBayes)
>>> help(limma.toptable)
"""