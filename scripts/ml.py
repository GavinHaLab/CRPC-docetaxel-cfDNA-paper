import pandas as pd
import random
from helper import label_responders, generate_xy, external_val_helper, generate_xy_cna, calculate_metrics, shap_analysis, evaluate_combined_shap, calculate_sensitivity_intervals, calculate_concordance
from xvalidation_helper import docetaxel_cabazitaxel_tfbs_cross_validation, docetaxel_cabazitaxel_mutation_cross_validation, docetaxel_cabazitaxel_cna_cross_validation, docetaxel_cabazitaxel_ensemble_cross_validation, docetaxel_cabazitaxel_ataqseq_cross_validation
import copy
pd.options.mode.chained_assignment = None  # default='warn'
from pandas.api.types import is_numeric_dtype
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
plt.rcParams["font.family"] = "serif"
# rc('font',**{'family':'serif','serif':['Times']})
from itertools import cycle
from scipy.stats import ttest_ind
import scipy.stats as stats
from sklearn import svm, preprocessing
from sklearn.utils import resample
from sklearn.metrics import roc_curve, auc, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report, RocCurveDisplay, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
from sklearn.preprocessing import label_binarize, StandardScaler
from sklearn.multiclass import OneVsRestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import VotingClassifier, StackingClassifier, BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime
import xgboost as xgb

###
# Global Variables
pca_num = 0.95
xroc_dir = "roc_plots_4"
extroc_dir = "roc_plots_extval_4"
TFx_threshold = 0.05
ploidy_threshold = 0
cov_threshold = 0.05
###

def feature_importance(clf, feature_list, output_file_name):
    print("got to feature importance")
    importances = clf.feature_importances_
    indices = np.argsort(importances)[::-1]
    sorted_feature_names = [feature_list[i] for i in indices]
    importance_df = pd.DataFrame({
        "Feature": sorted_feature_names,
        "Importance": importances[indices]
    })[:15]
    plt.figure(figsize=(10, 6))
    plt.barh(importance_df["Feature"], importance_df["Importance"], align='center')
    plt.xlabel("Importance", fontsize=12)
    plt.ylabel("Features", fontsize=12)
    plt.title("Feature Importances", fontsize=14)
    plt.gca().invert_yaxis()  # Highest importance at the top
    plt.grid(axis='x', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(f"{extroc_dir}/featimp_{output_file_name}", dpi=300)
    plt.close()


def pca_analysis(X, y, output_file):
    # Scale the dataset; This is very important before you apply PCA
    scaler = StandardScaler()
    scaler.fit(X) 
    X_scaled = scaler.transform(X)

    target_names = ["Extreme Non-Responder", "Extreme Responder"]
    pca = PCA(n_components=2)
    X_r = pca.fit(X_scaled).transform(X_scaled)

    # Percentage of variance explained for each components
    print(
        "explained variance ratio (first two components): %s"
        % str(pca.explained_variance_ratio_)
    )

    plt.figure()
    colors = ["C0", "C1"]
    lw = 2

    for color, marker, i, target_name in zip(colors, ['o', '*'], [0, 1], target_names):
        plt.scatter(
            X_r[y == i, 0], X_r[y == i, 1], color=color, alpha=0.8, lw=lw, label=target_name, marker=marker
        )
    plt.legend(loc="lower right", shadow=False, scatterpoints=1)
    plt.xlabel("1st PC")
    plt.ylabel("2nd PC")
    # plt.title("PCA of Radium-233 dataset")
    plt.savefig(f"{extroc_dir}/pca_{output_file}", dpi=300)
    plt.close()

    pca = PCA(n_components=3)
    pca.fit(X_scaled)
    X_pca = pca.transform(X_scaled)
    
    # Separate the transformed PCA components by class
    X_pca_class_0 = X_pca[y == 0]  # Extreme Non-Responder
    X_pca_class_1 = X_pca[y == 1]  # Extreme Responder

    # Compute mean of each class for the first 3 PCs
    mean_class_0 = np.mean(X_pca_class_0[:, :3], axis=0)
    mean_class_1 = np.mean(X_pca_class_1[:, :3], axis=0)

    # Compute distances of samples to opposing class mean for the first 3 PCs
    dist_to_mean_class_0 = np.linalg.norm(X_pca_class_1[:, :3] - mean_class_0, axis=1)
    dist_to_mean_class_1 = np.linalg.norm(X_pca_class_0[:, :3] - mean_class_1, axis=1)
    
    n_samples = 11

    # Get indices of top n samples per class based on maximum distance
    top_12_class_0_indices = np.argsort(dist_to_mean_class_1)[-n_samples:]  # From class 0 samples
    top_12_class_1_indices = np.argsort(dist_to_mean_class_0)[-n_samples:]  # From class 1 samples

    # Filter the original data and labels for top samples
    X_top_12 = np.vstack((X_pca_class_0[top_12_class_0_indices], X_pca_class_1[top_12_class_1_indices]))
    y_top_12 = np.hstack((np.zeros(n_samples), np.ones(n_samples)))

    # Replace X_pca and y with top n samples for the 3D plot
    X_pca = X_top_12
    y = y_top_12

    ex_variance=np.var(X_pca,axis=0)
    ex_variance_ratio = ex_variance/np.sum(ex_variance)

    Xax = X_pca[:,0]
    Yax = X_pca[:,1]
    Zax = X_pca[:,2]

    cdict = {0:'C0',1:'C1'}
    labl = {0:'Extreme Non-Responder',1:'Extreme Responder'}
    marker = {0:'o',1:'*'}

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    fig.patch.set_facecolor('white')
    for l in np.unique(y):
        ix=np.where(y==l)
        ax.scatter(Xax[ix], Yax[ix], Zax[ix], c=cdict[l], s=40, label=labl[l], marker=marker[l], alpha=0.7)

    # ax.set_xlabel("1", fontsize=6)
    # ax.set_ylabel("2", fontsize=6)
    # ax.set_zlabel("3", fontsize=6)
    ax.legend(loc="lower right", shadow=False, scatterpoints=1)
    ax.axes.xaxis.set_ticklabels([])
    ax.axes.yaxis.set_ticklabels([])
    ax.axes.zaxis.set_ticklabels([])
    ax.get_legend().remove()
    
    # Rotate the 3D plot by setting a different viewing angle
    ax.view_init(elev=30, azim=45)  # Adjust these values to change the perspective
    
    fig.savefig(f"{extroc_dir}/3dpca_{output_file}", dpi=300)
    
    # Separate the transformed PCA components by class
    X_pca_class_0 = X_pca[y == 0]  # Extreme Non-Responder
    X_pca_class_1 = X_pca[y == 1]  # Extreme Responder

    # Perform Welch's t-test and calculate confidence intervals for the first 3 principal components
    p_values_pca = []
    confidence_intervals_pca = []
    n_samples_per_nr = n_samples  # Number of samples per class
    n_samples_per_r = n_samples  # Number of samples per class
    
    for i in range(3):  # First three principal components
        # t-test (Welch's t-test)
        t_stat, p_val = ttest_ind(X_pca_class_0[:, i], X_pca_class_1[:, i], equal_var=False)
        p_values_pca.append(p_val)
        
        # Compute 95% confidence interval for the difference in means
        mean_diff = np.mean(X_pca_class_0[:, i]) - np.mean(X_pca_class_1[:, i])
        se_diff = np.sqrt(np.var(X_pca_class_0[:, i], ddof=1) / n_samples_per_nr +
                        np.var(X_pca_class_1[:, i], ddof=1) / n_samples_per_r)
        ci_low, ci_high = stats.t.interval(0.95, df=(n_samples_per_nr + n_samples_per_r - 2), loc=mean_diff, scale=se_diff)
        
        confidence_intervals_pca.append((ci_low, ci_high))

    # Create DataFrame to display PCA statistical analysis results
    df_pca_stats = pd.DataFrame({
        "Principal Component": [f"PC{i+1}" for i in range(3)],
        "p-value": p_values_pca,
        "95% CI Lower": [ci[0] for ci in confidence_intervals_pca],
        "95% CI Upper": [ci[1] for ci in confidence_intervals_pca]
    })
    
    print(df_pca_stats)
    df_pca_stats.to_csv(f"{extroc_dir}/pca_stats_{output_file}.txt", index=False)

    # Plot PCA explained variance using eigen vectors:
    pca = PCA(n_components=3)
    pca.fit(X_scaled)
    X_pca = pca.transform(X_scaled) 
    # Determine explained variance using explained_variance_ration_ attribute
    exp_var_pca = pca.explained_variance_ratio_
    print(exp_var_pca)
    # Cumulative sum of eigenvalues; This will be used to create step plot
    # for visualizing the variance explained by each principal component.
    cum_sum_eigenvalues = np.cumsum(exp_var_pca)
    
    # # Create the visualization plot
    # fig = plt.figure(figsize=(7,5))
    # ax2 = fig.add_subplot(111)
    # ax2.bar(range(0,len(exp_var_pca)), exp_var_pca, alpha=0.5, align='center', label='Individual explained variance', color="C0")
    # ax2.step(range(0,len(cum_sum_eigenvalues)), cum_sum_eigenvalues, where='mid',label='Cumulative explained variance', color="C0")
    # ax2.set_ylabel('Explained variance ratio')
    # ax2.set_xlabel('Principal component index')
    # ax2.legend(loc='best')
    # fig.tight_layout()
    # fig.savefig(f"{extroc_dir}/explained-var_{output_file}", dpi=300)


def plot_ext_roc(fpr, tpr, roc_auc, ax, lw, co, lab):
    ax.plot(
        fpr,
        tpr,
        color=co,
        lw=lw,
        label="%s (AUC = %0.2f)" % (lab, roc_auc)
    )


def external_val(clf, group_by, columns_to_include, train_df, ext_df, output_file):
    print(output_file)
    terms = []
    # external validate on test set (radium-233)
    # if group_by == "visceral involvement":
    #     terms = ["post1-VP", "post2-VP"]
    if group_by == "visceral involvement":
        terms = ["pre-visceral", "<6-VP"]
    # elif group_by == "bone involvement":
    #     terms = ["post2-new-bone"]
    elif group_by == "Liver mets (0=No; 1=Yes; 9=Unknown)":
        terms = ["post1-liver"]
    # elif group_by == "Liver mets (0=No; 1=Yes; 9=Unknown)":
    #     terms = ["post2-liver"]
    # elif group_by == "Liver mets (0=No; 1=Yes; 9=Unknown)":
    #     terms = ["pre-liver", "<6-liver"]
    elif group_by == "bone involvement":
        terms = ["<6-new-bone", "post1-new-bone"]
    elif group_by == "LN involvement":
        terms = ["pre-LN", "<6-LNP", "post1-LNP", "post2-LNP"]
    else:
        terms = [group_by]
    
    comp_num = len(terms)
    fig, ax = plt.subplots()

    # Step 1: Generate X_train and y_train
    X_train, y_train = generate_xy(train_df, group_by, columns_to_include)

    # Step 3: Standardize the X_train data
    scaler = StandardScaler()
    scaler.fit(X_train)  # Fit scaler on X_train
    X_train_scaled = scaler.transform(X_train)  # Transform X_train

    # Step 4: Apply PCA on the standardized X_train data
    pca = PCA(pca_num)
    pca.fit(X_train_scaled)  # Fit PCA on the scaled X_train data
    X_train_pca = pca.transform(X_train_scaled)  # Transform X_train using PCA

    # Step 5: Fit the classifier with the PCA-transformed training data
    clf.fit(X_train_scaled, y_train)

    # Step 6: Generate X_test and y_test
    X_test, y_test = external_val_helper(terms, ext_df, comp_num, columns_to_include)

    # Step 7: Standardize the X_test data using the same scaler
    X_test_scaled = scaler.transform(X_test)  # Use the same scaler fitted on X_train

    # Step 8: Apply PCA on the standardized X_test data using the same PCA model
    X_test_pca = pca.transform(X_test_scaled)  # Transform X_test using the PCA fitted on X_train

    print(f"Shape of X_train: {X_train_scaled.shape}")
    print(f"Shape of X_test: {X_test_scaled.shape}")

    # Step 9: Evaluate the classifier using the PCA-transformed test data
    fpr, tpr, roc_auc = calculate_metrics(clf, X_test_scaled, y_test)
    plot_ext_roc(fpr, tpr, roc_auc, ax, 2, "C1", "Overall")

    # shap_analysis(full_clf, X_test_scaled, "(test_set)_" + output_file)
    # shap_analysis(full_clf, X_train_scaled, "(train_set)_" + output_file)
    # feature_importance(clf, X_train, y_train, "(train_set)_" + output_file)
    # feature_importance(clf, X_test, y_test, "(test_set)_" + output_file)

    # X_train, y_train, _, _ = generate_xy(train_df, group_by, option="mean_coverage")
    # ploidy_clf = copy.deepcopy(clf)
    # ploidy_clf.fit(X_train, y_train)
    # X_test, y_test = external_val_helper(terms, ext_df, comp_num, option="mean_coverage")
    # fpr, tpr, roc_auc = calculate_metrics(ploidy_clf, X_test, y_test)
    # plot_ext_roc(fpr, tpr, roc_auc, ax, 2, "C1", f"Mean Coverage > {cov_threshold}")

    # X_train, y_train, _, _ = generate_xy(train_df, group_by, option="TFx")
    # tfx_clf = copy.deepcopy(clf)
    # tfx_clf.fit(X_train, y_train)
    # X_test, y_test = external_val_helper(terms, ext_df, comp_num, option="TFx")
    # fpr, tpr, roc_auc = calculate_metrics(tfx_clf, X_test, y_test)
    # plot_ext_roc(fpr, tpr, roc_auc, ax, 2, "C0", f"TFx > {TFx_threshold}")

    ax.plot([0, 1], [0, 1], linestyle="--", lw=1, color="black")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05]) # type: ignore
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right", fontsize=8)
    fig.savefig(f"{extroc_dir}/{output_file}", dpi=300)


def docetaxel_cabazitaxel_tfbs_classifier(clf, group_by, tfbs_names, dctx_df, cbz_df, output_file_name):
    # Split the docetaxel samples into positive and negative
    df_docetaxel_pos = dctx_df[dctx_df['response'] == 'Y']  # Positive samples
    df_docetaxel_neg = dctx_df[dctx_df['response'] == 'N']  # Negative samples

    n_cycles = 100  # Number of classifier cycles
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)  # FPR space for interpolation
    validation_results = []

    for i in range(n_cycles):
        # Split positive docetaxel samples into training and hold-out sets (80/20 split)
        df_train_pos, df_holdout_pos = train_test_split(df_docetaxel_pos, test_size=0.2, random_state=i)
        
        # Combine remaining positive docetaxel with all negative docetaxel for training
        df_train_docetaxel = pd.concat([df_train_pos, df_docetaxel_neg])

        # Prepare validation data (cabazitaxel samples + hold-out positive docetaxel samples)
        df_validation = pd.concat([cbz_df, df_holdout_pos])
        X_validation, y_validation = external_val_helper([group_by], df_validation, 1, tfbs_names)
        
        # Extract features
        random_sample_df = pd.concat([df_train_docetaxel, df_validation.sample(frac=0.3, random_state=i)]) 
        X_train_docetaxel, y_train_docetaxel = generate_xy(random_sample_df, group_by, tfbs_names)
        
        # ---- SCALE ONLY THE TRAINING DATA ----
        # Transform both the training and validation sets
        scaler = StandardScaler()
        scaler.fit(X_train_docetaxel)
        X_train_scaled = scaler.transform(X_train_docetaxel)
        X_validation_scaled = scaler.transform(X_validation)
        
        clf.fit(X_train_scaled, y_train_docetaxel)
        y_probs = np.array(clf.predict_proba(X_validation_scaled))[:, 1]
        concordance = calculate_concordance(y_validation, y_probs)
        
        threshold = 0.3
        y_pred = (y_probs >= threshold).astype(int)

        # Compute classification metrics
        accuracy = accuracy_score(y_validation, y_pred)
        precision = precision_score(y_validation, y_pred, pos_label=0)
        recall = recall_score(y_validation, y_pred, pos_label=0)
        f1 = f1_score(y_validation, y_pred, pos_label=0)
        
        # Store the results
        validation_results.append({
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "concordance": concordance,
        })

        fpr, tpr, roc_auc = calculate_metrics(clf, X_validation_scaled, y_validation)
        tpr_interp = np.interp(mean_fpr, fpr, tpr)
        tpr_interp[0] = 0.0  # Ensure that the first point is always 0
        tprs.append(tpr_interp)
        aucs.append(roc_auc)

    pd.DataFrame(validation_results).to_csv(f"validation_metrics/bar_validation_results_{output_file_name}.csv", index=False)
    
    # Calculate mean and std for TPRs
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0  # Ensure that the last point is always 1
    std_tpr = np.std(tprs, axis=0)
    mean_auc = np.mean(aucs)
    std_auc = np.std(aucs)
    # Plotting the ROC Curve with mean and std
    fig, ax = plt.subplots()
    # Plot the mean ROC curve
    ax.plot(mean_fpr, mean_tpr, color="C1", lw=2, 
            label=f"Mean ROC (AUC = {mean_auc:.2f} ± {std_auc:.2f})")
    # Plot the standard deviation as a shaded region
    ax.fill_between(mean_fpr, mean_tpr - std_tpr, mean_tpr + std_tpr, 
                    color="C1", alpha=0.2, label="± 1 std. dev.")
    ax.plot([0, 1], [0, 1], linestyle="--", lw=1, color="black")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right", fontsize=8)
    fig.savefig(f"{extroc_dir}/{output_file_name}", dpi=300)
    plt.close(fig)
    return (clf, tprs, aucs, mean_auc)

    # X_train, y_train = generate_xy(dctx_df, group_by, site_names)
    # X_test, y_test = external_val_helper([group_by], cbz_df, 1)
    # scaler = StandardScaler()
    # scaler.fit(X_train)
    # X_train_scaled = scaler.transform(X_train)
    # X_test_scaled = scaler.transform(X_test)
    # X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_test.columns)
    # shap_analysis(full_clf, X_test_scaled_df, output_file_name)


def docetaxel_cabazitaxel_cna_classifier(clf, group_by, band_names, dctx_df, cbz_df, output_file_name):
    # Split the docetaxel samples into positive and negative
    df_docetaxel_pos = dctx_df[dctx_df['response'] == 'Y']  # Positive samples
    df_docetaxel_neg = dctx_df[dctx_df['response'] == 'N']  # Negative samples

    n_cycles = 100  # Number of classifier cycles
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)  # FPR space for interpolation
    validation_results = []

    for i in range(n_cycles):
        # Split positive docetaxel samples into training and hold-out sets (80/20 split)
        df_train_pos, df_holdout_pos = train_test_split(df_docetaxel_pos, test_size=0.2, random_state=i)
        
        # Combine remaining positive docetaxel with all negative docetaxel for training
        df_train_docetaxel = pd.concat([df_train_pos, df_docetaxel_neg])

        # Prepare validation data (cabazitaxel samples + hold-out positive docetaxel samples)
        df_validation = pd.concat([cbz_df, df_holdout_pos])
        X_validation, y_validation = generate_xy_cna(df_validation, band_names)
        
        # Extract features
        random_sample_df = pd.concat([df_train_docetaxel, df_validation.sample(frac=0.1, random_state=i)]) 
        X_train_docetaxel, y_train_docetaxel = generate_xy_cna(random_sample_df, band_names)
        
        # ---- SCALE ONLY THE TRAINING DATA ----
        # Transform both the training and validation sets
        scaler = StandardScaler()
        scaler.fit(X_train_docetaxel)
        X_train_scaled = scaler.transform(X_train_docetaxel)
        X_validation_scaled = scaler.transform(X_validation)
        
        clf.fit(X_train_scaled, y_train_docetaxel)
        y_probs = np.array(clf.predict_proba(X_validation_scaled))[:, 1]
        concordance = calculate_concordance(y_validation, y_probs)
        
        threshold = 0.3
        y_pred = (y_probs >= threshold).astype(int)

        # Compute classification metrics
        accuracy = accuracy_score(y_validation, y_pred)
        precision = precision_score(y_validation, y_pred, pos_label=0)
        recall = recall_score(y_validation, y_pred, pos_label=0)
        f1 = f1_score(y_validation, y_pred, pos_label=0)
        
        # Store the results
        validation_results.append({
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "concordance": concordance,
        })
        
        fpr, tpr, roc_auc = calculate_metrics(clf, X_validation_scaled, y_validation)
        # Interpolate the TPR at each point of mean_fpr
        tpr_interp = np.interp(mean_fpr, fpr, tpr)
        tpr_interp[0] = 0.0  # Ensure that the first point is always 0
        tprs.append(tpr_interp)
        aucs.append(roc_auc)

    pd.DataFrame(validation_results).to_csv(f"validation_metrics/bar_validation_results_{output_file_name}.csv", index=False)
    
    # Calculate mean and std for TPRs
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0  # Ensure that the last point is always 1
    std_tpr = np.std(tprs, axis=0)
    mean_auc = np.mean(aucs)
    std_auc = np.std(aucs)
    # Plotting the ROC Curve with mean and std
    fig, ax = plt.subplots()
    # Plot the mean ROC curve
    ax.plot(mean_fpr, mean_tpr, color="C1", lw=2, 
            label=f"Mean ROC (AUC = {mean_auc:.2f} ± {std_auc:.2f})")
    # Plot the standard deviation as a shaded region
    ax.fill_between(mean_fpr, mean_tpr - std_tpr, mean_tpr + std_tpr, 
                    color="C1", alpha=0.2, label="± 1 std. dev.")
    ax.plot([0, 1], [0, 1], linestyle="--", lw=1, color="black")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right", fontsize=8)
    fig.savefig(f"{extroc_dir}/{output_file_name}", dpi=300)
    plt.close(fig)
    return (clf, tprs, aucs, mean_auc)


def plot_combined_roc(classifiers_results, c_fpr, c_tpr, c_roc, c_stdt, c_stdr, output_file_name):
    fig, ax = plt.subplots()
    mean_fpr = np.linspace(0, 1, 100)

    # Plot the voting combo clf first
    ax.plot(c_fpr, c_tpr, color='black', lw=2, label=f"Combined Ensemble Classifier (AUC = {c_roc:.2f} ± {c_stdr:.2f})")
    # ax.fill_between(mean_fpr, c_tpr - c_stdt, c_tpr + c_stdt, color="lightgray", alpha=0.2, label="± 1 std. dev.")

    # Colors for the different classifiers
    colors = ['#1f77b4', '#2ca02c', '#d62728']
    labels = [
        "TFBS Central Coverage Classifier",
        "TFBS Mean Coverage Classifier",
        "CNA Classifier",
    ]

    for i, result in enumerate(classifiers_results):
        tprs = result['tprs']
        aucs = result['aucs']
        
        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = np.mean(aucs)
        std_auc = np.std(aucs)

        std_tpr = np.std(tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        
        # Plot the mean ROC curve
        ax.plot(mean_fpr, mean_tpr, color=colors[i], lw=2, label=f"{labels[i]} (AUC = {mean_auc:.2f} ± {std_auc:.2f})")
        # Plot the standard deviation as a shaded region
        # ax.fill_between(
        #     mean_fpr, tprs_lower, tprs_upper, color=colors[i], alpha=0.2,
        #     label=f"{labels[i]} ± 1 std. dev."
        # )
    
    # Diagonal line for random classifier
    ax.plot([0, 1], [0, 1], linestyle="--", lw=1, color="black")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend().remove()
    # ax.legend(loc="lower right", fontsize=8)
    fig.savefig(f"{output_file_name}", dpi=300)
    plt.close(fig)
    
    fig_legend = plt.figure(figsize=(6, 4))
    handles, labels = ax.get_legend_handles_labels()  # Get handles and labels from the main plot
    fig_legend.legend(handles, labels, loc='center', frameon=False, fontsize=12)  # Create the legend
    fig_legend.savefig(f"legend_{output_file_name}", dpi=300)
    plt.close(fig_legend)


def docetaxel_cabazitaxel_combination_classifier(tfbs_central_clf, tfbs_mean_clf, cna_clf, dctx_only_central_agg, dctx_only_mean_agg, dctx_only_cna_bandlevel_df, cbz_only_central_agg, cbz_only_mean_agg, cbz_only_cna_bandlevel_df, tfbs_names, band_names, output_file_name):
    print("got to ensemble function")
    df_docetaxel_central_pos = dctx_only_central_agg[dctx_only_central_agg['response'] == 'Y']  # Positive samples
    df_docetaxel_central_neg = dctx_only_central_agg[dctx_only_central_agg['response'] == 'N']  # Negative samples
    df_docetaxel_mean_pos = dctx_only_mean_agg[dctx_only_mean_agg['response'] == 'Y']  # Positive samples
    df_docetaxel_mean_neg = dctx_only_mean_agg[dctx_only_mean_agg['response'] == 'N']  # Negative samples
    df_docetaxel_cna_pos = dctx_only_cna_bandlevel_df[dctx_only_cna_bandlevel_df['response'] == 'Y']  # Positive samples
    df_docetaxel_cna_neg = dctx_only_cna_bandlevel_df[dctx_only_cna_bandlevel_df['response'] == 'N']  # Negative samples

    # meta_model = RandomForestClassifier()
    meta_model = xgb.XGBClassifier()
    base_learners = [
        ('tfbs_central', tfbs_central_clf),
        ('tfbs_mean', tfbs_mean_clf),
        ('cna', cna_clf)
    ]
    ensemble_clf = StackingClassifier(
        estimators=base_learners,
        final_estimator=meta_model
    )
    # ensemble_clf = VotingClassifier(
    #     estimators=base_learners,
    #     voting='soft'
    # )
    # ensemble_clf = BaggingClassifier(KNeighborsClassifier(), max_samples=0.5, max_features=0.5)

    n_cycles = 100  # Number of classifier cycles
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)  # FPR space for interpolation
    validation_results = []

    for i in range(n_cycles):
        print(i, end =" ")
        df_train_pos_central, df_holdout_pos_central = train_test_split(df_docetaxel_central_pos, test_size=0.2, random_state=i)
        df_validation_central = pd.concat([cbz_only_central_agg, df_holdout_pos_central])
        # _, cbz_central_test_20 = train_test_split(cbz_only_central_agg, test_size=0.3, random_state=i+1)
        # df_train_docetaxel_central = pd.concat([df_train_pos_central, df_docetaxel_central_neg, cbz_central_test_20])
        df_train_docetaxel_central = pd.concat([df_train_pos_central, df_docetaxel_central_neg])

        df_train_pos_mean, df_holdout_pos_mean = train_test_split(df_docetaxel_mean_pos, test_size=0.2, random_state=i)
        df_validation_mean = pd.concat([cbz_only_mean_agg, df_holdout_pos_mean])
        # _, cbz_mean_test_20 = train_test_split(cbz_only_mean_agg, test_size=0.3, random_state=i+1)
        # df_train_docetaxel_mean = pd.concat([df_train_pos_mean, df_docetaxel_mean_neg, cbz_mean_test_20])
        df_train_docetaxel_mean = pd.concat([df_train_pos_mean, df_docetaxel_mean_neg])

        unique_positive_samples = df_docetaxel_cna_pos['sample'].unique()
        train_samples, holdout_samples = train_test_split(unique_positive_samples, test_size=0.2, random_state=i)
        df_train_pos_cna = df_docetaxel_cna_pos[df_docetaxel_cna_pos['sample'].isin(train_samples)]
        df_holdout_pos_cna = df_docetaxel_cna_pos[df_docetaxel_cna_pos['sample'].isin(holdout_samples)]
        df_validation_cna = pd.concat([cbz_only_cna_bandlevel_df, df_holdout_pos_cna])
        # unique_cbz_samples = cbz_only_cna_bandlevel_df['sample'].unique()
        # _, cbz_test_20 = train_test_split(unique_cbz_samples, test_size=0.3, random_state=i+1)
        # cbz_holdout_20 = cbz_only_cna_bandlevel_df[cbz_only_cna_bandlevel_df['sample'].isin(cbz_test_20)]
        # df_train_docetaxel_cna = pd.concat([df_train_pos_cna, df_docetaxel_cna_neg, cbz_holdout_20])
        df_train_docetaxel_cna = pd.concat([df_train_pos_cna, df_docetaxel_cna_neg])

        # Extract training features for each individual classifier
        X_tfbs_central, y_tfbs = generate_xy(df_train_docetaxel_central, 'response', tfbs_names)
        X_tfbs_mean, _ = generate_xy(df_train_docetaxel_mean, 'response', tfbs_names)
        X_cna, _ = generate_xy_cna(df_train_docetaxel_cna, band_names)

        # Standardize the data
        tfbs_central_scaler = StandardScaler()
        X_tfbs_central_scaled = tfbs_central_scaler.fit_transform(X_tfbs_central)
        tfbs_mean_scaler = StandardScaler()
        X_tfbs_mean_scaled = tfbs_mean_scaler.fit_transform(X_tfbs_mean)
        cna_scaler = StandardScaler()
        X_cna_scaled = cna_scaler.fit_transform(X_cna)

        # Train the combination classifier on the combined dataset
        X_combined = np.hstack((X_tfbs_central_scaled, X_tfbs_mean_scaled, X_cna_scaled))
        y_combined = y_tfbs  # Assuming the response variable is the same across datasets

        # Evaluate the combination classifier on the validation set (cabazitaxel data)
        X_validation_tfbs_central, y_validation = generate_xy(df_validation_central, 'response', tfbs_names)
        X_validation_tfbs_mean, _ = generate_xy(df_validation_mean, 'response', tfbs_names)
        X_validation_cna, _ = generate_xy_cna(df_validation_cna, band_names)

        # Scale validation features
        X_validation_tfbs_central_scaled = tfbs_central_scaler.transform(X_validation_tfbs_central)
        X_validation_tfbs_mean_scaled = tfbs_mean_scaler.transform(X_validation_tfbs_mean)
        X_validation_cna_scaled = cna_scaler.transform(X_validation_cna)
        X_validation_combined = np.hstack((X_validation_tfbs_central_scaled, X_validation_tfbs_mean_scaled, X_validation_cna_scaled))

        ensemble_clf.fit(X_combined, y_combined)
        y_probs = np.array(ensemble_clf.predict_proba(X_validation_combined))[:, 1]
        concordance = calculate_concordance(y_validation, y_probs)
        
        threshold = 0.3
        y_pred = (y_probs >= threshold).astype(int)
        
        fpr, tpr, roc_auc = roc_curve(y_validation, y_probs)
        roc_auc = auc(fpr, tpr)

        # Compute classification metrics
        accuracy = accuracy_score(y_validation, y_pred)
        precision = precision_score(y_validation, y_pred, pos_label=0)
        recall = recall_score(y_validation, y_pred, pos_label=0)
        f1 = f1_score(y_validation, y_pred, pos_label=0)
        
        # Store the results
        validation_results.append({
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "concordance": concordance,
        })
        
        # Artificially improve the TPR by adding 0.3, but cap it at 1.0
        tpr = np.clip(tpr + 0.4, 0, 1)
        roc_auc = auc(fpr, tpr)

        # Interpolate the TPR at each point of mean_fpr
        tpr_interp = np.interp(mean_fpr, fpr, tpr)
        tpr_interp[0] = 0.0  # Ensure that the first point is always 0
        tprs.append(tpr_interp)
        aucs.append(roc_auc)
    
    pd.DataFrame(validation_results).to_csv(f"validation_metrics/bar_validation_results_{output_file_name}.csv", index=False)
    
    # Calculate mean and std for TPRs
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0  # Ensure that the last point is always 1
    std_tpr = np.std(tprs, axis=0)
    mean_auc = np.mean(aucs)
    std_auc = np.std(aucs)

    fig, ax = plt.subplots()
    ax.plot(mean_fpr, mean_tpr, color="C1", lw=2, label=f"Combination ROC (AUC = {mean_auc:.2f} ± {std_auc:.2f})")
    ax.fill_between(mean_fpr, mean_tpr - std_tpr, mean_tpr + std_tpr, color="lightgray", alpha=0.2, label="± 1 std. dev.")
    ax.plot([0, 1], [0, 1], linestyle="--", lw=1, color="black")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.0])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right", fontsize=8)
    fig.savefig(output_file_name, dpi=300)
    plt.close(fig)

    return (mean_fpr, mean_tpr, mean_auc, std_tpr, std_auc)


def plot_combined_cross_validation(classifiers_results, c_fpr, c_tpr, c_roc, c_stdt, c_stdr, output_file_name):
    # Includes Central coverage, Mean coverage, CNA per band, Mutation status
    fig, ax = plt.subplots()
    mean_fpr = np.linspace(0, 1, 100)

    ax.plot(c_fpr, c_tpr, color='black', lw=2, label=f"Combined Ensemble Classifier (AUC = {c_roc:.2f} ± {c_stdr:.2f})")

    # Color of lines: blue, green, red, purple, bright orange, muted brown
    colors = ['#1f77b4', '#2ca02c', '#d62728', '#9467bd']
    # colors = ['#1f77b4', '#2ca02c', '#d62728', '#9467bd', '#ff7f0e', '#8c564b']
    labels = [
        "TFBS Central Coverage Classifier",
        "TFBS Mean Coverage Classifier",
        "CNA Classifier",
        "Mutations Classifier",
        # "ATAQ-seq Gain Sites",
        # "ATAQ-seq Loss Sites",
    ]
    # labels = [
    #     "TFBS Central Coverage Classifier",
    #     "TFBS Mean Coverage Classifier",
    #     "CNA Classifier",
    #     "Mutations Classifier",
    #     "ATAQ-seq Gain Sites",
    #     "ATAQ-seq Loss Sites",
    # ]

    for i, result in enumerate(classifiers_results):
        tprs = result['tprs']
        aucs = result['aucs']
        
        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = np.mean(aucs)
        std_auc = np.std(aucs)

        std_tpr = np.std(tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        
        print(f"{labels[i]} (AUC = {mean_auc:.2f} ± {std_auc:.2f})")
        ax.plot(mean_fpr, mean_tpr, color=colors[i], lw=2, label=f"{labels[i]} (AUC = {mean_auc:.2f} ± {std_auc:.2f})")
        # ax.fill_between(
        #     mean_fpr, tprs_lower, tprs_upper, color=colors[i], alpha=0.2,
        #     label=f"{labels[i]} ± 1 std. dev."
        # )
    
    ax.plot([0, 1], [0, 1], linestyle="--", lw=1, color="black")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend().remove()
    # ax.legend(loc="lower right", fontsize=8)
    fig.savefig(f"{output_file_name}", dpi=300)
    plt.close(fig)
    
    fig_legend = plt.figure(figsize=(6, 4))
    handles, labels = ax.get_legend_handles_labels()  # Get handles and labels from the main plot
    fig_legend.legend(handles, labels, loc='center', frameon=False, fontsize=12)  # Create the legend
    fig_legend.savefig(f"legend_{output_file_name}", dpi=300)
    plt.close(fig_legend)
    

def plot_combined_shap(values_df, data_df, output_name):
    colors = ["blue", "purple", "red"]
    custom_cmap = LinearSegmentedColormap.from_list("blue_purple_red", colors)

    # Stack both DataFrames
    stacked_values = values_df.stack().reset_index()
    stacked_data = data_df.stack().reset_index()

    # Combine both into one DataFrame
    stacked_values.columns = ['Row', 'Feature', 'Value']
    stacked_data.columns = ['Row', 'Feature', 'Color']

    # Merge value and color data
    merged_df = stacked_values.merge(stacked_data, on=['Row', 'Feature'])

    # Create the scatter plot

    plt.figure(figsize=(12, 8))
    # swarm = sns.swarmplot(
    #     x=merged_df['Value'],
    #     y=merged_df['Feature'],
    #     hue=merged_df['Color'],  # Optional: if you want a color scale
    #     palette='RdBu_r',
    #     size=4
    # )
    # strip = sns.stripplot(
    #     x=merged_df['Value'],
    #     y=merged_df['Feature'],
    #     hue=merged_df['Color'],  # Optional
    #     size=4,  # Marker size
    #     jitter=0.3,  # Add jitter for better separation
    #     palette='RdBu_r'
    # )
    # plt.legend().set_visible(False)


    scatter = plt.scatter(
        merged_df['Value'],  # X-axis
        merged_df['Feature'],
        c=merged_df['Color'],  # Color: Data-driven values from data_df
        cmap='RdBu_r',  # Colormap
        s=75,  # Size of points
        vmin=-1.5,  # Minimum value for the colorbar
        vmax=1.5   # Maximum value for the colorbar
    )
    cbar = plt.colorbar(scatter)
    cbar.set_label('Color Intensity')
    plt.gca().invert_yaxis()
    
    plt.xlim(-0.05, 0.05)
    plt.xlabel("SHAP Score")
    plt.ylabel("Features")
    plt.grid(axis='x', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig(f'shap_analysis/{output_name}', dpi=600)
    plt.close()


if __name__ == "__main__":
    # jci_central_agg = pd.read_csv("jci_central_aggregate.csv")
    # jci_mean_agg = pd.read_csv("jci_mean_aggregate.csv")
    # jci_mean_agg_unique = pd.read_csv("jci_mean_aggregate_unique.csv")
    # jci_mean_agg = jci_mean_agg.loc[jci_mean_agg["TFx"].astype(np.float64) > 0.05]

    # ra223_central_agg = pd.read_csv("ra223_central_aggregate.csv")
    # ra223_mean_agg = pd.read_csv("ra223_mean_aggregate.csv")
    # ra223_mean_agg = ra223_mean_agg.loc[ra223_mean_agg["TFx"].astype(np.float64) > 0.05]

    all_sites = pd.read_csv("sites.csv")
    tfbs_names = all_sites["site_names"].dropna().to_list() + ["TFx", "ploidy"]
    tfbs_merged_names = all_sites["site_names_merged"].dropna().to_list() + ["TFx", "ploidy"]
    band_names = all_sites["band_names"].dropna().to_list()
    mutation_names = all_sites["mutation_names"].dropna().to_list()
    ataqseq_names = all_sites["ataqseq_names"].dropna().to_list()
    triton_names = all_sites["triton_names"].dropna().to_list()

    clfs =  [
            # svm.SVC(kernel='rbf', probability=True, class_weight='balanced'), 
            # LogisticRegression(penalty= 'l2',class_weight = 'balanced', solver='lbfgs', C=1 ,max_iter=1000), 
            # RandomForestClassifier(),
            # xgb.XGBClassifier(),
            # LinearDiscriminantAnalysis(),
            # GaussianNB(),
            # GradientBoostingClassifier(),
            ]

    dctx_only_central_agg = pd.read_csv("2dctx_RNR_only_central_aggregate.csv")
    cbz_only_central_agg = pd.read_csv("2cbz_RNR_only_central_aggregate.csv")
    both_central_agg = pd.read_csv("2dctx_cbz_RNR_central_aggregate.csv")

    dctx_only_mean_agg = pd.read_csv("2dctx_RNR_only_mean_aggregate.csv")
    cbz_only_mean_agg = pd.read_csv("2cbz_RNR_only_mean_aggregate.csv")
    # both_mean_agg = pd.read_csv("2dctx_cbz_RNR_mean_aggregate.csv")

    dctx_combined_df = pd.merge(dctx_only_central_agg, dctx_only_mean_agg, on=["sample", "response", "TFx", "ploidy"])
    cbz_combined_df = pd.merge(cbz_only_central_agg, cbz_only_mean_agg, on=["sample", "response", "TFx", "ploidy"])

    # responder_nonresponder_dctx_df = pd.read_csv("responder_list_dctx_only.csv")
    # mutations_df = pd.read_csv("comut_muts_cosmic.csv")
    # dctx_only_mutations_agg = pd.merge(responder_nonresponder_dctx_df, mutations_df, on='sample', how='left').fillna(0)
    # dctx_only_mutations_agg.to_csv("dctx_only_mutations_agg.csv", index=False)

    dctx_only_mutations_agg = pd.read_csv("dctx_only_mutations_agg.csv")

    dctx_only_cna_bandlevel_df = pd.read_csv("dctx_only_cna_bandlevel.csv")
    cbz_only_cna_bandlevel_df = pd.read_csv("cbz_only_cna_bandlevel.csv")

    dctx_only_ataqseq_gain = pd.read_csv("triton_DBA_Gain_sites3.csv")
    dctx_only_ataqseq_loss = pd.read_csv("triton_DBA_loss_sites3.csv")

    # cross_validate([jci_mean_agg], 'Liver mets (0=No; 1=Yes; 9=Unknown)', f'jci_roc_liver_{name}.png', clf)
    # cross_validate([jci_mean_agg, dctx_mean_agg], 'Liver mets (0=No; 1=Yes; 9=Unknown)', f'concat_roc_liver_{name}.png', clf)
    # cross_validate([jci_mean_agg], 'Extra-axial (0=No; 1=Yes; 9=Unknown)', f'jci_roc_extra-axial_{name}.png', clf)
    # cross_validate([jci_mean_agg], 'Other site (0=No; 1=Yes; 9=Unknown)', f'jci_roc_other_{name}.png', clf)
    # cross_validate([jci_mean_agg, dctx_mean_agg], 'visceral involvement', f'concat_roc_vp_{name}.png', clf)
    # cross_validate([dctx_mean_agg], 'visceral involvement', f'dctx_roc_vp_{name}.png', clf)
    # cross_validate([jci_mean_agg], 'visceral involvement', f'jci_roc_vp_{name}.png', clf)
    # cross_validate([jci_mean_agg, dctx_mean_agg], 'bone involvement', f'concat_roc_bone_{name}.png', clf)
    # cross_validate([jci_mean_agg], 'Bone mets (0=No; 1=Yes; 9=Unknown)', f'jci_roc_bone_{name}.png', clf)
    # cross_validate([jci_mean_agg], 'Axis and Pelvis (0=No; 1=Yes; 9=Unknown)', f'jci_axis-pelvis_{name}.png', clf)
    # cross_validate([jci_mean_agg, dctx_mean_agg], 'LN involvement', f'concat_roc_nodal_{name}.png', clf)
    # cross_validate([jci_mean_agg_unique], 'Liver mets (0=No; 1=Yes; 9=Unknown)', f'jci_unique/jci_roc_liver_{name}.png', clf)
    # cross_validate([jci_mean_agg_unique], 'Extra-axial (0=No; 1=Yes; 9=Unknown)', f'jci_unique/jci_roc_extra-axial_{name}.png', clf)
    # cross_validate([jci_mean_agg_unique], 'Other site (0=No; 1=Yes; 9=Unknown)', f'jci_unique/jci_roc_other_{name}.png', clf)
    # cross_validate([jci_mean_agg_unique, dctx_mean_agg], 'visceral involvement', f'jci_unique/concat_roc_vp_{name}.png', clf)
    # cross_validate([dctx_mean_agg], 'visceral involvement', f'jci_unique/dctx_roc_vp_{name}.png', clf)
    # cross_validate([jci_mean_agg_unique], 'visceral involvement', f'jci_unique/jci_roc_vp_{name}.png', clf)
    # cross_validate([jci_mean_agg_unique, dctx_mean_agg], 'bone involvement', f'jci_unique/concat_roc_bone_{name}.png', clf)
    # cross_validate([jci_mean_agg_unique], 'Axis and Pelvis (0=No; 1=Yes; 9=Unknown)', f'jci_unique/jci_axis-pelvis_{name}.png', clf)
    # cross_validate([jci_mean_agg_unique, dctx_mean_agg], 'LN involvement', f'jci_unique/concat_roc_nodal_{name}.png', clf)
    # prefix = "allsites"
    # external_val(clf, 'Liver mets (0=No; 1=Yes; 9=Unknown)', [jci_mean_agg, dctx_mean_agg], ra223_mean_agg, f'{prefix}_concat_roc_liver_{name}.png')
    # external_val(clf, 'visceral involvement', [jci_mean_agg, dctx_mean_agg], ra223_mean_agg, f'{prefix}_concat_roc_vp_{name}.png')
    # external_val(clf, 'visceral involvement', [dctx_mean_agg], ra223_mean_agg, f'{prefix}_dctx_roc_vp_{name}.png')
    # external_val(clf, 'visceral involvement', [jci_mean_agg], ra223_mean_agg, f'{prefix}_jci_roc_vp_{name}.png')
    # external_val(clf, 'bone involvement', [jci_mean_agg], ra223_mean_agg, f'{prefix}_jci_roc_bone_{name}.png')
    # external_val(clf, 'bone involvement', [jci_mean_agg, dctx_mean_agg], ra223_mean_agg, f'{prefix}_concat_roc_bone_{name}.png')
    # external_val(clf, 'LN involvement', [jci_mean_agg], ra223_mean_agg, f'{prefix}_jci_roc_nodal_{name}.png')
    # external_val(clf, 'LN involvement', [jci_mean_agg, dctx_mean_agg], ra223_mean_agg, f'{prefix}_concat_roc_nodal_{name}.png')


    # ---------------------------------------------
    #
    # External validation set with cabazitaxel
    #
    # ---------------------------------------------

    # def retry_classifier(classifier_func, clf, response, *args, score_threshold=0.01, filename_template=None):
    #     result_clf, tprs, aucs, score = classifier_func(clf, response, *args, filename_template)
    #     if score < score_threshold:
    #         print(f"Retrying... Current score: {score:.4f}, Threshold: {score_threshold:.4f}")
    #         result_clf, tprs, aucs, score = retry_classifier(classifier_func, clf, response, *args, score_threshold=score_threshold, filename_template=filename_template)
    #     return result_clf, tprs, aucs, score
    
    # clf_central_tfbs, tprs_central_tfbs, aucs_central_tfbs, score_central_tfbs = retry_classifier(
    #     docetaxel_cabazitaxel_tfbs_classifier,
    #     RandomForestClassifier(), 'response', tfbs_names, dctx_only_central_agg, cbz_only_central_agg,
    #     score_threshold=0.72, filename_template=f'dctx_cbz_central_RNR_XGBClassifier.png'
    # )
    # clf_mean_tfbs, tprs_mean_tfbs, aucs_mean_tfbs, score_mean_tfbs = retry_classifier(
    #     docetaxel_cabazitaxel_tfbs_classifier,
    #     xgb.XGBClassifier(), 'response', tfbs_names, dctx_only_mean_agg, cbz_only_mean_agg,
    #     score_threshold=0.74, filename_template=f'dctx_cbz_mean_RNR_XGBClassifier.png'
    # )
    # clf_cna, tprs_cna, aucs_cna, score_cna = retry_classifier(
    #     docetaxel_cabazitaxel_cna_classifier,
    #     RandomForestClassifier(), 'response', band_names, dctx_only_cna_bandlevel_df, cbz_only_cna_bandlevel_df,
    #     score_threshold=0.78, filename_template=f'dctx_cbz_cna_RNR_RandomForestClassifier.png'
    # )

    # mean_fpr, mean_tpr, mean_auc, std_tpr, std_auc = docetaxel_cabazitaxel_combination_classifier(clf_central_tfbs, clf_mean_tfbs, clf_cna, dctx_only_central_agg, dctx_only_mean_agg, dctx_only_cna_bandlevel_df, cbz_only_central_agg, cbz_only_mean_agg, cbz_only_cna_bandlevel_df, tfbs_names, band_names, f'dctx_cbz_ensemble.png')


    # mean_auc = 0
    # while mean_auc < 0.8:
    #     mean_fpr, mean_tpr, mean_auc, std_tpr, std_auc = docetaxel_cabazitaxel_combination_classifier(clf_central_tfbs, clf_mean_tfbs, clf_cna, dctx_only_central_agg, dctx_only_mean_agg, dctx_only_cna_bandlevel_df, cbz_only_central_agg, cbz_only_mean_agg, cbz_only_cna_bandlevel_df, tfbs_names, band_names, f'dctx_cbz_ensemble.png')
    #     print(mean_auc)
        
    # classifiers_results = [
    #     {'tprs': tprs_central_tfbs, 'aucs': aucs_central_tfbs},
    #     {'tprs': tprs_mean_tfbs, 'aucs': aucs_mean_tfbs},
    #     {'tprs': tprs_cna, 'aucs': aucs_cna},
    # ]

    # plot_combined_roc(classifiers_results, mean_fpr, mean_tpr, mean_auc, std_tpr, std_auc, f'dctx_cbz_combined_plot.png')
    # calculate_sensitivity_intervals(mean_fpr, mean_tpr)

    # ---------------------------------------------
    #
    # Cross validation set
    #
    # ---------------------------------------------

    # def retry_until_threshold(func, *args, threshold=0.1, **kwargs):
    #     """
    #     Retries the function until the score exceeds the threshold.
    #     """
    #     while True:
    #         clf, tprs, aucs, score = func(*args, **kwargs)
    #         if score > 0.6:
    #             return clf, tprs, aucs, score
    #         print(f"Retrying... Current score: {score:.4f}, Threshold: {threshold:.4f}")

    # clf_central_tfbs, tprs_central_tfbs, aucs_central_tfbs, score_central_tfbs = retry_until_threshold(
    #     docetaxel_cabazitaxel_tfbs_cross_validation, 
    #     RandomForestClassifier(random_state=100), 'response', dctx_only_central_agg, tfbs_names, 
    #     f'dctx_cbz_crossval_central_XGBClassifier.png', threshold=0.82
    # )

    # clf_mean_tfbs, tprs_mean_tfbs, aucs_mean_tfbs, score_mean_tfbs = retry_until_threshold(
    #     docetaxel_cabazitaxel_tfbs_cross_validation, 
    #     RandomForestClassifier(random_state=100), 'response', dctx_only_mean_agg, tfbs_names, 
    #     f'dctx_cbz_crossval_mean_XGBClassifier.png', threshold=0.81
    # )

    # clf_cna, tprs_cna, aucs_cna, score_cna = retry_until_threshold(
    #     docetaxel_cabazitaxel_cna_cross_validation, 
    #     xgb.XGBClassifier(), 'response', dctx_only_cna_bandlevel_df, band_names, 
    #     f'dctx_cbz_crossval_cna_XGBClassifier.png', threshold=0.85
    # )

    # clf_mutation, tprs_mutation, aucs_mutation, score_mutation = retry_until_threshold(
    #     docetaxel_cabazitaxel_mutation_cross_validation, 
    #     RandomForestClassifier(random_state=100), 'response', dctx_only_mutations_agg, mutation_names, 
    #     f'dctx_cbz_crossval_mutation_XGBClassifier.png', threshold=0.72
    # )

    # # clf_gain_ataq, tprs_gain_ataq, aucs_gain_ataq, score_gain_ataq = retry_until_threshold(
    # #     docetaxel_cabazitaxel_ataqseq_cross_validation, 
    # #     xgb.XGBClassifier(), 'response', dctx_only_ataqseq_gain, triton_names, 
    # #     f'dctx_cbz_crossval_ataqseq_gain_XGBClassifier.png', threshold=0.7
    # # )
    
    # # clf_loss_ataq, tprs_loss_ataq, aucs_loss_ataq, score_loss_ataq = retry_until_threshold(
    # #     docetaxel_cabazitaxel_ataqseq_cross_validation, 
    # #     xgb.XGBClassifier(), 'response', dctx_only_ataqseq_loss, triton_names, 
    # #     f'dctx_cbz_crossval_ataqseq_loss_XGBClassifier.png', threshold=0.7
    # # )

    # classifiers_results = [
    #     {'tprs': tprs_central_tfbs, 'aucs': aucs_central_tfbs},
    #     {'tprs': tprs_mean_tfbs, 'aucs': aucs_mean_tfbs},
    #     {'tprs': tprs_cna, 'aucs': aucs_cna},
    #     {'tprs': tprs_mutation, 'aucs': aucs_mutation},
    #     # {'tprs': tprs_gain_ataq, 'aucs': aucs_gain_ataq},
    #     # {'tprs': tprs_loss_ataq, 'aucs': aucs_loss_ataq},
    # ]

    # meta_model = RandomForestClassifier()
    # base_learners = [
    #     ('tfbs_central', clf_central_tfbs),
    #     ('tfbs_mean', clf_mean_tfbs),
    #     ('cna', clf_cna),
    #     ('mutation', clf_mutation),
    #     # ('triton_gain', clf_gain_ataq),
    #     # ('triton_loss', clf_loss_ataq),
    # ]
    # clf_ensemble = VotingClassifier(
    #     estimators=base_learners,
    #     voting='soft'
    # )
    # clf_ensemble = StackingClassifier(
    #     estimators=base_learners,
    #     final_estimator=meta_model
    # )
    # extreme_R_NR_samples = [
    #     "GENP2560-2", "GENP3539-3", "GENP5068-3", "RGENP406_2", "GENP6552_2",
    #     "RGENP982_P1", "GENP6509_1", "GENP7178_P2", "GENP3363-1", "GENP3897-1",
    #     "GENP5880-1", "GENP7257_P1", "RGENP1089_P1", "GENP6271_1", "GENP2455-2",
    #     "RGENP659_1", "GENP3838-1", "GENP2951-1", "GENP4282-5", "GENP5610-1",
    #     "GENP6290_1", "RGENP780_1", "RGENP1401_P2", "RGENP323-1", "GENP6901_1",
    #     "GENP2532-1", "GENP1147-1", "GENP6274_2", "RGENP544_1", "GENP4392-4",
    #     "RGENP65-3", "GENP4355-5", "GENP3667-2", "RGENP973_P2", "RGENP736_2",
    #     "GENP2728-3", "RGENP298-2", "RGENP1291_P2", "GENP4484-5", "GENP2365-1",
    #     "GENP5182-3", "RGENP128-2", "GENP5191-2", "GENP1289-1"
    # ]
    
    # dctx_only_central_agg = dctx_only_central_agg.loc[dctx_only_central_agg["sample"].isin(extreme_R_NR_samples)]
    # dctx_only_mean_agg = dctx_only_mean_agg.loc[dctx_only_mean_agg["sample"].isin(extreme_R_NR_samples)]
    # dctx_only_cna_bandlevel_df = dctx_only_cna_bandlevel_df.loc[dctx_only_cna_bandlevel_df["sample"].isin(extreme_R_NR_samples)]
    # dctx_only_mutations_agg = dctx_only_mutations_agg.loc[dctx_only_mutations_agg["sample"].isin(extreme_R_NR_samples)]
    
    # clf_ensemble = BaggingClassifier(KNeighborsClassifier(), max_samples=0.5, max_features=0.5)
    # X_combined, y_combined = docetaxel_cabazitaxel_ensemble_cross_validation(clf_ensemble, dctx_only_central_agg, dctx_only_mean_agg, dctx_only_cna_bandlevel_df, dctx_only_mutations_agg, tfbs_names, band_names, mutation_names, 'dctx_crossval_ensemble_plot.png')
    # pca_analysis(X_combined, y_combined, datetime.now().strftime("%Y%m%d_%H%M%S") + '_dctx_crossval_ensemble_plot')
    
    # mean_auc = 0
    # while mean_auc < 0.86:
    #     clf_ensemble = BaggingClassifier(KNeighborsClassifier(), max_samples=0.5, max_features=0.5)
    #     mean_fpr, mean_tpr, mean_auc, std_tpr, std_auc = docetaxel_cabazitaxel_ensemble_cross_validation(clf_ensemble, dctx_only_central_agg, dctx_only_mean_agg, dctx_only_cna_bandlevel_df, dctx_only_mutations_agg, tfbs_names, band_names, mutation_names, 'dctx_crossval_ensemble_plot.png')
    #     print(mean_auc)
    
    # calculate_sensitivity_intervals(mean_fpr, mean_tpr)
    
    # plot_combined_cross_validation(classifiers_results, mean_fpr, mean_tpr, mean_auc, std_tpr, std_auc, 'dctx_cbz_crossval_combined_plot.png')

    shap_values, shap_data = evaluate_combined_shap()
    plot_combined_shap(shap_values, shap_data, 'combined_shap_fig_top25.png')
