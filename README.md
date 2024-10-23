# Research_SAM


## Identify pairlevel biomarkers from TCGA to predict and treat cancer metastasis
+ Repository of source codes which reproduce the research article "Unveiling the novel pairwise biomarker to predict and treat cancer metastasis", Lim et al


## Requirements
+ python ( and v 2.7.13)
+ pandas (v 0.24.2)
+ numpy (v 1.16.6)
+ matplotlib (v 2.0.0)
+ scipy (v 1.2.2)
+ sklearn (v 0.20.2)
+ lifelines (v 0.19.5)
+ gseapy


## Installation
+ All python packages can be installed via pip or conda-forge
+ e.g. pip install lifelines
+ e.g. conda install conda-forge::lifelines
+ Link (PIP) : https://pypi.org/project/pip/
+ Link (Conda-forge) : https://anaconda.org/conda-forge/repo/


## Code (Python)
+ "run_ssGSEA.py" to generate pathway level expression profiles using single sample GSEA (ssGSEA) tool (gseapy)
+ "single_pathway_prediction.py" to predict drug response in cancer patients using a single pathway
+ "multiple_pathway_prediction.py" to predict drug response in cancer patients using multiple pathways

## Code (R)
+ "" to generate gene expression profile using limma


## Network proximity code
+ We calculated network proximity in PPI network by using codes from 'Network-based in silico drug efficacy screening' Emre et al, Nature Communications, 2016
+ Link : https://github.com/emreg00/toolbox
