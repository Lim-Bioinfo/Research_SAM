import shap
import warnings
import random
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import gzip

from combat.pycombat import pycombat
from functools import reduce
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, roc_auc_score, average_precision_score
from matplotlib import pyplot as plt
from sklearn import set_config
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler
from imblearn.ensemble import BalancedRandomForestClassifier


if __name__=="__main__":
    print("Open file")
    sam_df = pd.read_csv("../Result/All_SAM_pairs.csv") #Data from Supplementary Data S2
    biomarker_df = pd.read_excel("../Input/Biomarkers_metastasis_references.xlsx") #Data from Supplementary Data S9
    clinical_df = pd.read_excel("../Input/Clinical_Meta_PM.xlsx") #Data from Supplementary Data S5
    df1 = pd.read_csv("../Input/GSE7553_series_matrix_collapsed_to_symbols.gct", sep = "\t")
    df2 = pd.read_csv("../Input/GSE8401_series_matrix_collapsed_to_symbols.gct", sep = "\t")
    df3 = pd.read_csv("../Input/GSE15605_series_matrix_collapsed_to_symbols.gct", sep = "\t")
    df4 = pd.read_csv("../Input/GSE46517_series_matrix_collapsed_to_symbols.gct", sep = "\t")
    df5 = pd.read_csv("GSE65904_series_matrix_collapsed_to_symbols.gct", sep = "\t")

    # Data preprocessing
    del df1['DESCRIPTION'];del df2['DESCRIPTION'];del df3['DESCRIPTION'];del df4['DESCRIPTION'];del df5['DESCRIPTION']
    df1 = df1.set_index(keys='NAME');df2 = df2.set_index(keys='NAME');df3 = df3.set_index(keys='NAME');df4 = df4.set_index(keys='NAME');df5 = df5.set_index(keys='NAME')
    df1_list = list(df1.columns);df2_list = list(df2.columns);df3_list = list(df3.columns);df4_list = list(df4.columns);df5_list = list(df5.columns)           
    df1_copied = df1.copy();df2_copied = df2.copy();df3_copied = df3.copy();df4_copied = df4.copy();df5_copied = df5.copy()

    df1_copied = np.log2(df1_copied + 1)
    df2_copied = np.log2(df2_copied + 1)
    df4_copied = np.log2(df4_copied + 1)
    df5_copied = np.log2(df5_copied + 1)
                                 
    data_frames = [df1_copied, df2_copied, df3_copied, df4_copied, df5_copied]
    df_merged = reduce(lambda left, right: pd.merge(left, right, on=['NAME'], how='inner'), data_frames)

    batches = ['GSE7553']*len(df1.columns) + ['GSE8401']*len(df2.columns) + ['GSE15605']*len(df3.columns) + ['GSE46517']*len(df4.columns) + ['GSE65904']*len(df5.columns)
    df_corrected = pycombat(df_merged, batches)
    #df_corrected.to_csv("Meta_SKCM_PM_expression_revised.txt", sep = "\t")

    # Genes in SKCM SAM pairs
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

    # Metastasis or metastatic tumor related biomarker in SKCM from previous literatures
    candidate_dict = {'SAM genes' : list(set(df_corrected.index) & selected_genes)}

    for i in range(0, len(biomarker_df.index)):
        if biomarker_df.iloc[i, 1] in set(df_corrected.index):
            if biomarker_df.iloc[i, 0] not in candidate_dict.keys():
                candidate_dict[biomarker_df.iloc[i, 0]] = [biomarker_df.iloc[i, 1]]
                continue
            else:
                candidate_dict[biomarker_df.iloc[i, 0]].append(biomarker_df.iloc[i, 1])
                continue      

    label_dict = {'Primary' : list(clinical_df[clinical_df['Sample type'] == 'Primary']['Sample ID']), 
              'Metastases' :list(clinical_df[clinical_df['Sample type'] == 'Metastases']['Sample ID'])}
    pm_status = [clinical_df.iloc[i, 0] for i in range(0, len(clinical_df)) if clinical_df.iloc[i, 4] != 'Unclassified']
    
    #Predict tumor types with various metastaasis-related biomarkers
    loo = LeaveOneOut()
    fpr_tpr = []
    aucs = []

    for selected_genes_list in sorted(candidate_dict.keys()):
        all_true, all_pred, all_prob = [], [], []

        df_corrected_T = pd.DataFrame(StandardScaler().fit_transform(df_corrected.values.T).T, columns=df_corrected.columns, index=df_corrected.index)
        df_corrected_T = df_corrected_T[pm_status]
        df_selected = df_corrected_transposed[sorted(candidate_dict[selected_genes_list])]

        X = df_selected  # Features
        sample_label_map = {sample: 1 if sample in label_dict['Metastases'] else 0 for sample in df_selected.index}
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

        auroc = roc_auc_score(all_true, all_prob)
        auprc = average_precision_score(all_true, all_prob)
        accuracy = accuracy_score(all_true, all_pred)
        precision = precision_score(all_true, all_pred)
        recall = recall_score(all_true, all_pred)
        f1 = f1_score(all_true, all_pred)
        fpr, tpr, _ = roc_curve(all_true, all_prob)
        fpr_tpr.append((fpr, tpr))
        aucs.append(roc_auc)
    
        print(f"Gene list: {selected_genes_list}")
        print(f"AUROC: {roc_auc}")
        print(f"AUPRC: {roc_auc}")
        print(f"Accuracy: {accuracy}")
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print(f"F1 Score: {f1}")
        print('--------------')
    

    # Plot AUC curves with distinct colors
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
    X_all = df_corrected.loc[:, pm_status]     # genes x samples

    # Restrict to candidate genes that exist
    genes = sorted(set(candidate_dict['SAM genes']) & set(X_all.index))
    if not genes:
        raise ValueError("No overlap between candidate SAM genes and expression matrix.")

    # Enforce fixed order and transpose to samples x genes
    X_all = X_all.loc[genes]                   # genes x samples
    X_all_T = X_all.T.copy()                   # samples x genes

    y = X_all_T.index.to_series().map(lambda s: 1 if s in label_dict['Metastases'] else 0).astype(int).values

    # LOOCV and SHAP (no scaling for trees)
    loo = LeaveOneOut()
    n, p = X_all_T.shape
    all_shap = np.zeros((n, p))
    heldout_prob = np.full(n, np.nan)

    for _, (tr, te) in enumerate(tqdm(loo.split(X_all_T), total=n, desc="LOOCV")):
        if np.unique(y[tr]).size < 2: # Both classes needs to training
            continue

        # Fresh clone; fit on TRAIN only
        BRF.fit(X_all_T.iloc[tr], y[tr])

        # Held-out probability (QC / AUC)
        heldout_prob[te] = BRF.predict_proba(X_all_T.iloc[te])[:, 1]

        # SHAP with TreeExplainer; background = training data
        explainer = shap.TreeExplainer(BRF, data=X_all_T.iloc[tr], model_output="probability")
        shap_vals = explainer.shap_values(X_all_T.iloc[te])  # may be list or ndarray

        # Robustly extract positive-class SHAP of shape (1, p)
        if isinstance(shap_vals, (list, tuple)):
            # list-of-classes API
            pos_idx = int(np.where(brf.classes_ == 1)[0][0])
            sv_pos = shap_vals[pos_idx]                                # (1, p)
        else:
            arr = np.asarray(shap_vals)
            if arr.ndim == 3 and arr.shape[-1] == 2:
                # condensed array with class axis at the end: (n_test, p, 2)
                pos_idx = int(np.where(brf.classes_ == 1)[0][0])
                sv_pos = arr[..., pos_idx]                             # (1, p)
            elif arr.ndim == 2 and arr.shape[1] == p:
                # already (n_test, p)
                sv_pos = arr
            else:
                raise ValueError(f"Unexpected SHAP shape: {arr.shape}")

        all_shap[te, :] = np.asarray(sv_pos).reshape(1, -1)


    # Aggregate out-of-sample SHAP and performance
    mean_abs = np.nanmean(np.abs(all_shap), axis=0)    # ranking metric
    mean_signed = np.nanmean(all_shap, axis=0)         # direction (toward Metastases if >0)

    feature_importances = (pd.DataFrame({"Feature": genes,"mean_abs_SHAP": mean_abs,"mean_signed_SHAP": mean_signed}).sort_values("mean_abs_SHAP", ascending=False).reset_index(drop=True))

    print("\nTop features by mean(|SHAP|):")
    print(feature_importances.head(30))

    # AUROC results from training using LOOCV (only where predictions exist and both classes present)
    mask = ~np.isnan(heldout_prob)
    if mask.sum() >= 2 and len(np.unique(y[mask])) == 2:
        auc = roc_auc_score(y[mask], heldout_prob[mask])
        print(f"\nLOOCV AUC = {auc:.3f}  (n evaluated = {mask.sum()}/{n})")
    else:
        print("\nLOOCV AUC not computed (insufficient evaluated folds or single-class labels).")


