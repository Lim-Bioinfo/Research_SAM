Identification pairlevel biomarkers from large-cohort of cancer patients to predict and treat cancer metastasis
Repository of source codes which reproduce the research article "Unveiling the novel pairwise biomarker to predict and treat cancer metastasis", Lim et al

Requirements
python ( and v 2.7.13)
pandas (v 0.24.2)
matplotlib (v 2.0.0)
numpy (v 1.16.6)
scipy (v 1.2.2)
sklearn (v 0.20.2)
lifelines (v 0.19.5)
gseapy

Installation instruction
All python packages can be installed via pip (https://pypi.org/project/pip/) or conda-forge (https://anaconda.org/conda-forge/repo)
e.g. pip install lifelines
e.g. conda install conda-forge::lifelines
Installations would take few minutes for each python package.

Code (Python)
"run_ssGSEA.py" to generate pathway level expression profiles using single sample GSEA (ssGSEA) tool (gseapy)
"single_pathway_prediction.py" to predict drug response in cancer patients using a single pathway
"multiple_pathway_prediction.py" to predict drug response in cancer patients using multiple pathways

Code (R)


We used network proximity calculation by using codes from 'Network-based in silico drug efficacy screening' Emre et al, Nature Communications, 2016
https://github.com/emreg00/toolbox
