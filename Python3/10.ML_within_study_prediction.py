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
    warnings.filterwarnings('ignore')
    
    print("Open file")
    sam_df = pd.read_csv("../Result/All_SAM_pairs.csv")
    biomarker_df = pd.read_excel("../Input/Biomarkers_metastasis_references.xlsx")
    clinical_df = pd.read_excel("../Input/Clinical_Meta_PM.xlsx") #Data from Supplementary Data S2
    df1 = pd.read_csv("../Input/GSE7553_series_matrix_collapsed_to_symbols.gct", sep = "\t")
    df2 = pd.read_csv("../Input/GSE8401_series_matrix_collapsed_to_symbols.gct", sep = "\t")
    df3 = pd.read_csv("../Input/GSE15605_series_matrix_collapsed_to_symbols.gct", sep = "\t")
    df4 = pd.read_csv("../Input/GSE46517_series_matrix_collapsed_to_symbols.gct", sep = "\t")

    # Batch correction
    label = {'GSE7553':{'Primary':list(clinical_df[(clinical_df['Sample type'] == 'Primary') & clinical_df['Dataset'] == 'GSE7553']), 
                      'Metastases':list(clinical_df[(clinical_df['Sample type'] == 'Metastases') & clinical_df['Dataset'] == 'GSE7553'])}
            'GSE8401':{'Primary':list(clinical_df[(clinical_df['Sample type'] == 'Primary') & clinical_df['Dataset'] == 'GSE8401']), 
                      'Metastases':list(clinical_df[(clinical_df['Sample type'] == 'Metastases') & clinical_df['Dataset'] == 'GSE8401'])}
            'GSE15605':{'Primary':list(clinical_df[(clinical_df['Sample type'] == 'Primary') & clinical_df['Dataset'] == 'GSE15605']), 
                      'Metastases':list(clinical_df[(clinical_df['Sample type'] == 'Metastases') & clinical_df['Dataset'] == 'GSE15605'])}
            'GSE46517':{'Primary':list(clinical_df[(clinical_df['Sample type'] == 'Primary') & clinical_df['Dataset'] == 'GSE46517']), 
                      'Metastases':list(clinical_df[(clinical_df['Sample type'] == 'Metastases') & clinical_df['Dataset'] == 'GSE46517'])}}


    del df1['DESCRIPTION'];del df2['DESCRIPTION'];del df3['DESCRIPTION'];del df4['DESCRIPTION']
    df1 = df1.set_index(keys='NAME');df2 = df2.set_index(keys='NAME');df3 = df3.set_index(keys='NAME');df4 = df4.set_index(keys='NAME')
    df1_list = list(df1.columns);df2_list = list(df2.columns);df3_list = list(df3.columns);df4_list = list(df4.columns)             
    df1_copied = df1.copy();df2_copied = df2.copy();df3_copied = df3.copy();df4_copied = df4.copy()

    df1_copied = np.log2(df1_copied + 1)
    df2_copied = np.log2(df2_copied + 1)
    df4_copied = np.log2(df4_copied + 1)
                                 
    data_frames = [df1_copied, df2_copied, df3_copied, df4_copied]
    df_merged = reduce(lambda left, right: pd.merge(left, right, on=['NAME'], how='inner'), data_frames)

    batches = ['GSE7553']*len(df1.columns) + ['GSE8401']*len(df2.columns) + ['GSE15605']*len(df3.columns) + ['GSE46517']*len(df4.columns)
    df_corrected = pycombat(df_merged, batches)
    #df_corrected.to_csv("GSE_merged_batchcorrected_sv.gct", sep = "\t")

    primary = label['GSE7553']['Primary'] + label['GSE8401']['Primary'] + label['GSE15605']['Primary'] + label['GSE46517']['Primary']
    metastases = label['GSE7553']['Metastases'] + label['GSE8401']['Metastases'] + label['GSE15605']['Metastases'] + label['GSE46517']['Metastases']
    label_dict = {'Primary' : primary, 'Metastases' : metastases}

    ## Genes in SKCM SAM pairs
    sam_dict = dict()
    for i in range(0, len(sam_df.index)):
        if sam_df.iloc[i, 0] not in sam_dict.keys():
            sam_dict[sam_df.iloc[i, 0]] = [tuple(sorted((sam_df.iloc[i, 1], sam_df.iloc[i, 2])))]
            continue
        else:
            sam_dict[sam_df.iloc[i, 0]].append(tuple(sorted((sam_df.iloc[i, 1], sam_df.iloc[i, 2]))))
            continue 

    input_pair = sam_dict['SKCM']
    selected_genes = set()
    for pair in input_pair:
        selected_genes.add(pair[0])
        selected_genes.add(pair[1])

    ## Metastasis or metastatic tumor related biomarker in SKCM from previous literatures
    candidate_dict = {'SAM genes' : list(set(df_corrected.index) & selected_genes)}

    for i in range(0, len(biomarker_df.index)):
        if biomarker_df.iloc[i, 1] in set(df_corrected.index):
            if biomarker_df.iloc[i, 0] not in candidate_dict.keys():
                candidate_dict[biomarker_df.iloc[i, 0]] = [biomarker_df.iloc[i, 1]]
                continue
            else:
                candidate_dict[biomarker_df.iloc[i, 0]].append(biomarker_df.iloc[i, 1])
                continue      

    ## Predict tumor types with various metastaasis-related biomarkers
    loo = LeaveOneOut()
    fpr_tpr = []
    aucs = []

    for selected_genes_list in sorted(candidate_dict.keys()):
        all_true, all_pred, all_prob = [], [], []

        df_corrected_T = pd.DataFrame(StandardScaler().fit_transform(df_corrected.values.T).T, columns=df_corrected.columns, index=df_corrected.index)
        df_corrected_T = df_corrected_T[sorted(primary + metastases)]
        df_selected = df_corrected_transposed[sorted(candidate_dict[selected_genes_list])]

        X = df_selected  # Features
        sample_label_map = {sample: 1 if sample in label_dict['Primary'] else 0 for sample in df_selected.index}
        y = [sample_label_map[sample] for sample in X.index]

        # Define Classifiers with specific random states where applicable
         BRF = BalancedRandomForestClassifier(random_state=452456, max_features="log2", bootstrap=True, sampling_strategy = 'majority', replacement = True, class_weight = 'balanced_subsample')

        # Execute the LOOCV
        for train_index, test_index in loo.split(X):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = np.array(y)[train_index], np.array(y)[test_index]

            BRF.fit(X_train, y_train)
            y_pred = BRF.predict(X_test)
            probabilities = BRF.predict_proba(X_test)[:, 1]
        
            all_true.extend(y_test)
            all_pred.extend(y_pred)
            all_prob.extend(probabilities)

        roc_auc = roc_auc_score(all_true, all_prob)
        accuracy = accuracy_score(all_true, all_pred)
        precision = precision_score(all_true, all_pred)
        recall = recall_score(all_true, all_pred)
        f1 = f1_score(all_true, all_pred)
        fpr, tpr, _ = roc_curve(all_true, all_prob)
        fpr_tpr.append((fpr, tpr))
        aucs.append(roc_auc)
    
        print(f"Gene list: {selected_genes_list}")
        print(f"Accuracy: {accuracy}")
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print(f"F1 Score: {f1}")
        print(f"AUC: {roc_auc}")
        print('--------------')
    

    ## Plot AUC curves with distinct colors
    colors = ['#FF4D26', '#FF8847', '#F9BC66', '#548235', '#70AD47', '#A9D18E', '#2F5597', '#0070C0', '#9DC3E6', '#664BA0', '#B756D8', '#9A96FF']
    plt.figure(figsize=(12, 12))
    for i, color in zip(range(len(fpr_tpr)), colors):
        fpr, tpr = fpr_tpr[i]
        plt.plot(fpr, tpr, lw=2, label=f'ROC curve (area = {aucs[i]:0.3f}) for {sorted(candidate_dict.keys())[i]}', color=color)

    plt.plot([0, 1], [0, 1], color='black', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    #plt.title('Receiver Operating Characteristic for Different Models')
    plt.legend(loc="lower right")
    plt.savefig("../Result/ML performance.svg", dpi = 600)
    plt.show()


    ## Calculate mean SHAP values
    warnings.filterwarnings('ignore')
    
    # Prepare data
    df_corrected_transposed = df_corrected.T
    df_selected = df_corrected_transposed[sorted(candidate_dict['SAM genes'])]
    X = df_selected  # Features
    sample_label_map = {sample: 1 if sample in label_dict['Primary'] else 0 for sample in X.index}
    y = np.array([sample_label_map[sample] for sample in X.index])
    
    # Define LOOCV procedure
    all_shap_values = np.zeros((len(X), X.shape[1]))
    
    # Execute LOOCV
    for train_index, test_index in tqdm(list(loo.split(X)), desc = 'Samples'):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Model Training
        BRF.fit(X_train, y_train)

        # SHAP Explainer with a reduced background dataset
        background = shap.sample(X_train, 50)  # Sample 50 instances for speed
        explainer = shap.KernelExplainer(BRF.predict_proba, background, link="logit")
        shap_values = explainer.shap_values(X_test)[1]  # Index 1 for positive class SHAP values

        all_shap_values[test_index] = shap_values  # Store SHAP values for the test index

    # Average SHAP values across all LOOCV iterations
    mean_shap_values = np.mean(all_shap_values, axis=0)

    # Create a DataFrame for feature importances
    feature_importances = pd.DataFrame({
        'Feature': X.columns,
        'SHAP Importance': mean_shap_values
    }).sort_values(by='SHAP Importance', ascending=False)

    # Print the feature importances
    print(feature_importances)
