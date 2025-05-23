# Input file


## Input files for Python3 code
+ TCGA MC3 project data: mc3.v0.2.8.PUBLIC.maf.gz (Downloaded from PanCanAtlas)
+ TCGA clinical data: TCGA-Clinical Data Resource (CDR) Outcome.xlsx (Preprocessed from original data downloaded from PanCanAtlas)
+ Gene phylogenetic profile data: JS Lee_Phylo_profile_Natcommm.csv (Converted yuval.phylogenetic.profile.RData from https://github.com/jooslee/ISLE/tree/main/data, originated from Yuval Tabach et al. Mol Syst Biol. (2013), Supplementary Table 1)
+ Phylogenetic weight data: JS Lee_Phylo_weight_Naturecomm.txt (Converted feature.weight.RData from https://github.com/jooslee/ISLE/tree/main/data, originated from Ensembl database)
+ Gene expression profile for TCGA cohort data: EBPlusPlusAdjustPANCAN_IlluminaHiSeq_RNASeqV2.geneExp.tsv (Downloaded from PanCanAtlas)
+ Human SL pair data: Human_SL_SynLethDB.csv (Downloaded from SynLethDB, v 2.0)
+ Experimentally identified SL pair data: JS Lee_NatComm_Experimentally identified gold standard SL interactions.csv (Originated from JS Lee et al, 2018, Nature comm, Supplementary Data 1)
+ Gebe essential profile data2: NatComm_Nichols et al_essesntial_profile.xlsx (Originated from Nichols et al, 2020, Nature comm, Supplementary Data 1)
+ Gene expression profile for SKCM from GEO: GSEXXXXX_series_matrix_collapsed_to_symbols.gct (Originated from GSEXXXXX_series_matrix.txt.gz and preprocessed by GSEA)
+ SKCM patient clinical data for Meta-Cohort PM from GEO: Clinical_Meta_PM.xlsx (Data from Supplementary Data S2 of this reseasrch, originated from GEO)
+ SKCM patient clinical data for Meta-Cohort S from GEO: Clinical_Meta_S.xlsx (Data from Supplementary Data S3 of this reseasrch, originated from GEO)
+ External somatic mutation data from cBioPortal, annotated by ANNOVAR: XXX_cBioPortal_annotated_mutation.tsv.gz (Originated from data_mutations.txt from Liu et al or Snyder et al from cBioPortal)
+ External clinical data: XXX_data_clinical_patient.txt, XXX_data_clinical_sample.txt (Originated from data_clinical_patient.txt and data_clinical_sample.txt from Liu et al or Snyder et al from cBioPortal)
+ Metastasis-associated biomarkers from previous studies: Biomarkers_metastasis_references.xlsx (Data from Supplementary Data S5 of this research, originated from each previous researches)

## Input files for Python2 code
+ Human protein-protein interaction network: 9606.protein.links.v12.0.txt.gz (Downloaded from STRING DB, v.12.0)
+ Human protein information network: 9606.protein.info.v12.0.txt.gz (Downloaded from STRING DB, v.12.0)

## Input files for R code
+ Gene expression profile for SKCM from GEO metacohort: GSE_merged_batchcorrected_XX.gct (pm for primary and metastasis information, sv for survival analysis)
