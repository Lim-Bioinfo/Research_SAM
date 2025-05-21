# Research_SAM


## Identify pairlevel biomarkers from TCGA to predict and treat cancer metastasis
+ Repository of source codes which reproduce the research article "Mutation-informed gene pairs to predict melanoma metastasis", Lim et al


## Requirements
### Requirements for python3
+ Python (v 3.9.16)
+ pandas (v 2.2.2)
+ numpy (v 1.26.4)
+ matplotlib (v 3.9.2)
+ networkx (v 3.2.1)
+ tqdm (v 4.66.4)
+ scipy (v 1.12.0)
+ sklearn (v 1.0.2)
+ lifelines (v 0.27.8)
+ more-itertools (v 10.3.0)
+ gseapy (v 1.1.0)
+ combat (v 0.3.3)

### Requirements for python2
+ Python (v 2.7.16)
+ pandas (v 0.24.2)
+ numpy (v 1.14.2)
+ networkx (v 2.2)
+ tqdm (v 4.63.0)

### Requirements for R
+ R (v 4.2.3)
+ limma (v 3.54.2)


## Installation
+ All python packages can be installed via pip or conda-forge
+ e.g. pip install lifelines
+ e.g. conda install conda-forge::lifelines
+ Link (PIP) : https://pypi.org/project/pip/
+ Link (Conda-forge) : https://anaconda.org/conda-forge/repo/


## Network proximity code
+ We calculated network proximity in PPI network by using codes from 'Network-based in silico drug efficacy screening' Emre et al, Nature Communications, 2016
+ Link : https://github.com/emreg00/toolbox
