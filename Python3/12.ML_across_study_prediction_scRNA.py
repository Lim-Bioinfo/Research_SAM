import shap
import warnings
import random
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import gzip

from combat.pycombat import pycombat
from functools import reduce
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, roc_auc_score
from matplotlib import pyplot as plt
from sklearn import set_config
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler
from imblearn.ensemble import BalancedRandomForestClassifier

if __name__=="__main__":
    print("Open file")
    sam_df = pd.read_csv("../Result/All_SAM_pairs.csv")
    biomarker_df = pd.read_excel("../Input/Biomarkers_metastasis_references.xlsx")
    clinical_df = pd.read_excel("../Input/Clinical_Meta_PM.xlsx") #Data from Supplementary Data S2
    df1 = pd.read_csv("../Input/GSE7553_series_matrix_collapsed_to_symbols.gct", sep = "\t")
    df2 = pd.read_csv("../Input/GSE8401_series_matrix_collapsed_to_symbols.gct", sep = "\t")
    df3 = pd.read_csv("../Input/GSE15605_series_matrix_collapsed_to_symbols.gct", sep = "\t")
    df4 = pd.read_csv("../Input/GSE46517_series_matrix_collapsed_to_symbols.gct", sep = "\t")
    df11 = pd.read_csv("../Input/GSE81383_data_melanoma_scRNAseq_BT_2015-07-02.txt", sep = "\t")

    del df1['DESCRIPTION'];del df2['DESCRIPTION'];del df3['DESCRIPTION'];del df4['DESCRIPTION']
    df1 = df1.set_index(keys='NAME');df2 = df2.set_index(keys='NAME');df3 = df3.set_index(keys='NAME');df4 = df4.set_index(keys='NAME');df11 = df11.set_index(keys='NAME')
    df1_list = list(df1.columns);df2_list = list(df2.columns);df3_list = list(df3.columns);df4_list = list(df4.columns)   ;df11_list = list(df11.columns)             
  
    df1_copied = df1.copy();df2_copied = df2.copy();df3_copied = df3.copy();df4_copied = df4.copy();df11_copied = df11.copy()

    df1_copied = np.log2(df1_copied+1)
    df2_copied = np.log2(df2_copied+1)
    df4_copied = np.log2(df4_copied+1)
    df11_copied = np.log2(df11_copied+1)

    data_frames = [df1_copied, df2_copied, df3_copied, df4_copied, df11_copied]
    df_merged3 = reduce(lambda left, right: pd.merge(left, right, on = ["NAME"], how='inner'), data_frames)

    #label_list = sorted(list(clinical_df10[(clinical_df10['Tumor stage'] == 'Primary') | (clinical_df10['Tumor stage'] == 'Metastases')]['Sample ID']))
  
    batches3 = ['GSE7553']*len(df1.columns) + ['GSE8401']*len(df2.columns) + ['GSE15605']*len(df3.columns) + ['GSE46517']*len(df4.columns) + ['GSE81383']*len(df11.columns)
    df_corrected3 = pycombat(df_merged3, batches3)

    sam_dict = dict()
    for i in range(0, len(sam_df.index)):
        if sam_df.iloc[i, 0] not in sam_dict.keys():
            sam_dict[sam_df.iloc[i, 0]] = [tuple(sorted((sam_df.iloc[i, 1], sam_df.iloc[i, 2])))]
            continue
        else:
            sam_dict[sam_df.iloc[i, 0]].append(tuple(sorted((sam_df.iloc[i, 1], sam_df.iloc[i, 2]))))
            continue 

    selected_genes = set()
    for pair in sam_dict['SKCM']:
        selected_genes.add(pair[0])
        selected_genes.add(pair[1])

    candidate_dict = dict()
    candidate_dict['SAM genes'] = sorted(selected_genes & set(df_corrected3.index))

    result_ml_dict = dict()

    for selected_genes_list in sorted(candidate_dict.keys()):

        df_corrected_T = pd.DataFrame(StandardScaler().fit_transform(df_corrected3.values.T).T, columns=df_corrected3.columns, index=df_corrected3.index)
        df_corrected_transposed = df_corrected_T.T
    
        common_genes = set(df_corrected_transposed.columns)
        df_selected = df_corrected_transposed[sorted(set(candidate_dict[selected_genes_list]) & common_genes)]

        X_train = df_selected.T[pm_status].T  # Features
        X_test = df_selected.T[df11_list].T
    
        sample_label_map = {sample: 1 if sample in label_dict['Metastases'] else 0 for sample in X_train.index}
        Y_train = [sample_label_map[sample] for sample in X_train.index]

        # Define Classifiers with specific random states where applicable
        BRF = BalancedRandomForestClassifier(random_state=452456, max_features="log2", bootstrap=True, sampling_strategy = 'majority', replacement = True, class_weight = 'balanced_subsample')
        BRF.fit(X_train, Y_train)
        Y_temp_pred = BRF.predict(X_test)

        Y_pred = ["Metastases" if label == 1 else "Primary" for label in Y_temp_pred]
    
        result_ml_dict[selected_genes_list] = pd.DataFrame({"Sample ID" : df11_list, "Predicted" : Y_pred})
