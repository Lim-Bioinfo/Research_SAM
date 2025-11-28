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

def cox_regression_analysis(clinical_data):
    results = {}

    #cox_data = clinical_data.drop(columns=['gender'])
    clinical_data = clinical_data.copy()
    cox_data_dummies = pd.get_dummies(clinical_data[['SEX', 'AGE (Initial diagnosis)', 'OS (Weeks)', 'event', 'Predicted']])
        
    control_group_name = 'Predicted_Metastases'
    
    if control_group_name in cox_data_dummies.columns:
        cox_data_dummies.drop([control_group_name], axis=1, inplace=True)        
        
    cph = CoxPHFitter(penalizer=0.01)
    cph.fit(cox_data_dummies, duration_col='OS (Weeks)', event_col='event')
    return cph.summary

if __name__=="__main__":
    print("Open file")
    sam_df = pd.read_csv("../Result/All_SAM_pairs.csv")
    biomarker_df = pd.read_excel("../Input/Biomarkers_metastasis_references.xlsx") #Data from Supplementary Data S9
    clinical_df0 = pd.read_excel("../Input/Clinical_Meta_PM.xlsx") #Data from Supplementary Data S5

    # Train dataset, Meta-PM, preprocessing
    df1 = pd.read_csv("../Input/GSE7553_series_matrix_collapsed_to_symbols.gct", sep = "\t")
    df2 = pd.read_csv("../Input/GSE8401_series_matrix_collapsed_to_symbols.gct", sep = "\t")
    df3 = pd.read_csv("../Input/GSE15605_series_matrix_collapsed_to_symbols.gct", sep = "\t")
    df4 = pd.read_csv("../Input/GSE46517_series_matrix_collapsed_to_symbols.gct", sep = "\t")
    df5 = pd.read_csv("GSE65904_series_matrix_collapsed_to_symbols.gct", sep = "\t")

    del df1['DESCRIPTION'];del df2['DESCRIPTION'];del df3['DESCRIPTION'];del df4['DESCRIPTION'];del df5['DESCRIPTION']
    df1 = df1.set_index(keys='NAME');df2 = df2.set_index(keys='NAME');df3 = df3.set_index(keys='NAME');df4 = df4.set_index(keys='NAME');df5 = df5.set_index(keys='NAME')
    df1_list = list(df1.columns);df2_list = list(df2.columns);df3_list = list(df3.columns);df4_list = list(df4.columns);df5_list = list(df5.columns)           
    df1_copied = df1.copy();df2_copied = df2.copy();df3_copied = df3.copy();df4_copied = df4.copy();df5_copied = df5.copy()          
  

    df1_copied = np.log2(df1_copied+1)
    df2_copied = np.log2(df2_copied+1)
    df4_copied = np.log2(df4_copied+1)
    df5_copied = np.log2(df5_copied + 1)

    data_frames0 = [df1_copied, df2_copied, df3_copied, df4_copied, df5_copied]
    df_merged0 = reduce(lambda left, right: pd.merge(left, right, on=['NAME'], how='inner'), data_frames0)
    batches0 = ['GSE7553']*len(df1.columns) + ['GSE8401']*len(df2.columns) + ['GSE15605']*len(df3.columns) + ['GSE46517']*len(df4.columns) + ['GSE65904']*len(df5.columns)
    df_corrected0 = pycombat(df_merged0, batches0)
    df_corrected0 = pd.DataFrame(StandardScaler().fit_transform(df_corrected0.values.T).T, index=df_corrected0.index, columns=df_corrected0.columns)

    # Test dataset, Meta-SV, preprocessing
    clinical_df0 = pd.read_excel("../Input/Clinical_Meta_SV.xlsx") #Data from Supplementary Data S4
    df6 = pd.read_csv("GSE22153_series_matrix_collapsed_to_symbols.gct", sep = "\t")
    df7 = pd.read_csv("GSE22154_series_matrix_collapsed_to_symbols.gct", sep = "\t")
    df8 = pd.read_csv("GSE54467_series_matrix_collapsed_to_symbols.gct", sep = "\t")
    df9 = pd.read_csv("GSE59455_series_matrix_collapsed_to_symbols.gct", sep = "\t")
    df0 = pd.read_csv("TCGA_SKCM_median_expression_improved.txt", sep = "\t")

    del df6['DESCRIPTION'];del df7['DESCRIPTION'];del df8['DESCRIPTION'];del df9['DESCRIPTION']
    df6 = df5.set_index(keys='NAME');df7 = df7.set_index(keys='NAME');df8 = df8.set_index(keys='NAME');df9 = df9.set_index(keys='NAME');df10 = df10.set_index(keys='NAME')
    df6_list = list(df6.columns);df7_list = list(df7.columns);df8_list = list(df8.columns);df9_list = list(df9.columns);df10_list = list(df10.columns)            
    df6_copied = df6.copy();df7_copied = df7.copy();df8_copied = df8.copy();df9_copied = df9.copy();df10_copied = df10.copy()

    shift_amount = 1 - df6_copied.min().min()  # Ensure the min value is at least 1
    df6_copied_shifted = df6_copied + shift_amount
    df6_copied = np.log2(df6_copied_shifted+1)
    df7_copied = np.log2(df7_copied+1)
    df9_copied = np.log2(df9_copied+1)

    data_frames1 = [df6_copied, df7_copied, df8_copied, df9_copied, df10_copied]
    df_merged1 = reduce(lambda left, right: pd.merge(left, right, on=['NAME'], how='inner'), data_frames1)
    batches1 = ['GSE22153']*len(df6_copied.columns) + ['GSE22154']*len(df7_copied.columns) + ['GSE54467']*len(df8_copied.columns) + ['GSE59455']*len(df9_copied.columns)+ ['TCGA']*len(df10_copied.columns)
    df_corrected1 = pycombat(df_merged1, batches1)
    df_corrected1 = pd.DataFrame(StandardScaler().fit_transform(df_corrected1.values.T).T, index=df_corrected1.index, columns=df_corrected1.columns)

    # Prepare biomarkers
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

    # Metastasis or metastatic tumor related biomarker in SKCM from previous literatures
    candidate_dict = {'SAM genes' : list(set(df_corrected0.index) & set(df_corrected1.index) & selected_genes)}

    for i in range(0, len(biomarker_df.index)):
        if biomarker_df.iloc[i, 1] in set(df_corrected.index):
            if biomarker_df.iloc[i, 0] not in candidate_dict.keys():
                candidate_dict[biomarker_df.iloc[i, 0]] = [biomarker_df.iloc[i, 1]]
                continue
            else:
                candidate_dict[biomarker_df.iloc[i, 0]].append(biomarker_df.iloc[i, 1])
                continue  

    result_ml_dict = dict()

    for selected_genes_list in sorted(candidate_dict.keys()):
        temp_result_df = clinical_df1[['Sample ID', 'SEX' , 'AGE (Initial diagnosis)', 'OS (Weeks)', 'STATUS', 'SAM group']].copy()

        temp_result_df = temp_result_df[temp_result_df['Sample ID'].isin(df_corrected1.columns)].copy()
        common_genes = set(df_corrected0.index) & set(df_corrected1.index)
        df_selected0 = df_corrected0.T[sorted(set(candidate_dict[selected_genes_list]) & common_genes)].T
        df_selected1 = df_corrected1.T[sorted(set(candidate_dict[selected_genes_list]) & common_genes)].T

        print(selected_genes_list, len(df_selected0.index))

        X_train = df_selected0[pm_status].T  # Features
        X_test = df_selected1.T

        sample_label_map = {sample: 1 if sample in label_dict['Metastases'] else 0 for sample in X_train.index}
        Y_train = [sample_label_map[sample] for sample in X_train.index]

        # Define Classifiers with specific random states where applicable             
        BRF = BalancedRandomForestClassifier(random_state=452456, max_features="log2", bootstrap=True, sampling_strategy = 'majority', replacement = True, class_weight = 'balanced_subsample')
        BRF.fit(X_train, Y_train)
        Y_temp_pred = BRF.predict(X_test)

    
        Y_pred = ["Metastases" if label == 1 else "Primary" for label in Y_temp_pred]
        temp_result_df['Predicted'] = list(Y_pred)

        result_ml_dict[selected_genes_list] = temp_result_df


    # Predict survival prognosis of predicted primary tumor samples
    result_sv_dict = dict()

    for result in result_ml_dict.keys():
        try:
            temp_result_df = result_ml_dict[result]
            temp_result_df['event'] = temp_result_df['STATUS'].apply(lambda x: True if x == 'Deceased' else False)
    
            cancer_clinical_data = temp_result_df[['Sample ID', 'SEX', 'AGE (Initial diagnosis)', 'OS (Weeks)', 'event', 'Predicted']].dropna()
            cancer_clinical_data = cancer_clinical_data.drop_duplicates(ignore_index = True)
        
            result_sv_dict[result] = cox_regression_analysis(cancer_clinical_data)
        except:
            continue

    for result in result_sv_dict.keys():
        print(result)
        if 'Predicted_Primary' in result_sv_dict[result].index:
            print("HR: %s" % result_sv_dict[result].loc['Predicted_Primary', 'exp(coef)'])
            print("Lower: %s" % result_sv_dict[result].loc['Predicted_Primary', 'exp(coef) lower 95%'])
            print("Upper: %s" % result_sv_dict[result].loc['Predicted_Primary', 'exp(coef) upper 95%'])
            print("P-value: %s" % result_sv_dict[result].loc['Predicted_Primary', 'p'])
        print('-----')
