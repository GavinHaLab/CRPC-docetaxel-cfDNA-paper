import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.preprocessing import StandardScaler
import shap


def scale_column(column, target_min, target_max):
    original_min = column.min()
    original_max = column.max()
    normalized = (column - original_min) / (original_max - original_min)
    return target_min + normalized * (target_max - target_min)


def create_unique_df(jci_mean_agg):
    concat_list = []
    unique_samples = set()
    for i, con1 in jci_mean_agg.iterrows():
        patient = con1["patient"]
        if not patient in unique_samples:
            unique_samples.add(patient)
            df = jci_mean_agg[jci_mean_agg["patient"] == patient]
            concat_list.append(df[df.TFx == df.TFx.max()])
    unique_jci = pd.concat(concat_list)
    unique_jci.to_csv("jci_mean_aggregate_unique.csv")


def plot_cos():
    # Parameters
    a = 3  # Depth of the dip
    k = 1    # Width of the dip
    x = np.linspace(-20, 20, 1000)
    y = np.cos(x) - a * np.exp(-k * x**2)

    plt.figure(figsize=(20, 7))
    plt.plot(x, y)
    plt.fill_between(x, y, y2=y.min(), color='lightblue', alpha=0.5)  # Fill under the curve
    plt.savefig(f"cos_draw.png", dpi=300)


def label_responders(row):
    if 'dctx' in row and pd.notna(row['dctx']):
        if row['response'] == 'Y' and row['dctx'] == 'Y':
            return 'Y'
        else:
            return 'N'
    else:
        return row['response']


def generate_xy(df, group_by, columns_to_include):
    feature_snip = df[[group_by] + ["sample"] + columns_to_include].dropna()
    feature_snip.loc[feature_snip[group_by] == "N", group_by] = 0
    feature_snip.loc[feature_snip[group_by] == "Y", group_by] = 1
    feature_snip["class"] = np.where(((feature_snip[group_by].astype(str) == 'Y') | (feature_snip[group_by].astype(int) == 1)), 1, 0)
    X = feature_snip.set_index('sample')[columns_to_include]
    y = feature_snip.set_index('sample')["class"]
    return (X, y)


def external_val_helper(terms, ext_df, comp_num, columns_to_include):
    feature_snip = ext_df[terms + columns_to_include].dropna()
    for t in terms:
        feature_snip.loc[feature_snip[t] == "N", t] = 0
        feature_snip.loc[feature_snip[t].isnull(), t] = 0
        feature_snip.loc[feature_snip[t] == "Y", t] = 1

    if comp_num == 4:
        feature_snip.loc[:, ("class")] = np.where(((feature_snip[terms[0]].astype(int) == 1) | (feature_snip[terms[1]].astype(int) == 1) | (feature_snip[terms[2]].astype(int) == 1) | (feature_snip[terms[3]].astype(int) == 1)), 1, 0)
    elif comp_num == 3:
        feature_snip.loc[:, ("class")] = np.where(((feature_snip[terms[0]].astype(int) == 1) | (feature_snip[terms[1]].astype(int) == 1) | (feature_snip[terms[2]].astype(int) == 1)), 1, 0)
    elif comp_num == 2:
        feature_snip.loc[:, ("class")] = np.where(((feature_snip[terms[0]].astype(int) == 1) | (feature_snip[terms[1]].astype(int) == 1)), 1, 0)
    elif comp_num == 1:
        feature_snip.loc[:, ("class")] = np.where(((feature_snip[terms[0]].astype(int) == 1)), 1, 0)

    X_test = feature_snip[columns_to_include]
    y_test = feature_snip["class"]

    return (X_test, y_test)


def generate_xy_cna(df, band_names):
    # # Filter out rows where 'logR_Copy_Number' is NaN
    # df = df.dropna(subset=['logR_Copy_Number'])
    value_to_use = 'logR_Copy_Number'

    # Replace NaN in 'logR_Copy_Number' with the corresponding value from 'Corrected_Copy_Number'
    df['logR_Copy_Number'] = df['logR_Copy_Number'].fillna(df['Corrected_Copy_Number'])
    # Pivoting the DataFrame to create a matrix with samples as rows and chrBandName as columns
    X = df.pivot_table(values=value_to_use, index='sample', columns='chrBandName', fill_value=0)
    # Generating labels (y)
    y = df.drop_duplicates(subset='sample')['response'].apply(lambda x: 1 if x == 'Y' else 0).reset_index(drop=True)

    return (X, y)


def calculate_metrics(clf, X_test, y_test):
    probs = clf.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, probs)
    roc_auc = roc_auc_score(y_test, probs)
    return (fpr, tpr, roc_auc)


def shap_analysis(clf, X, feature_list, output_file):
    shap_dir = "shap_analysis"
    explainer = shap.TreeExplainer(clf, X, check_additivity=False)
    shap_obj = explainer(X)
    print(f"SHAP values shape: {shap_obj.values.shape}")
    print(f"Input data shape: {X.shape}")

    # Select SHAP values for the second class (e.g., class 1 aka 'positive' class)
    shap_values_class_1 = shap_obj.values[..., 1]  # Shape: (n_samples, n_features)
    shap_values_class_1 = np.nan_to_num(shap_values_class_1)  # Replace NaNs/infinities with 0
    shap_data_class_1 = shap_obj.data
    shap_data_class_1 = np.nan_to_num(shap_data_class_1)

    pd.DataFrame(
        shap_values_class_1,
        columns=feature_list
    ).to_csv(f"{shap_dir}/{output_file}_values.csv", index=False)
    pd.DataFrame(
        shap_data_class_1,
        columns=feature_list
    ).to_csv(f"{shap_dir}/{output_file}_data.csv", index=False)

    # plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_obj[..., 1], X, feature_names=feature_list, show=False)
    plt.savefig(f"{shap_dir}/shap_summary_{output_file}.png", dpi=300, bbox_inches='tight')
    plt.close()


def evaluate_combined_shap():
    central_data = pd.read_csv("shap_analysis/shap_central_data.csv").add_prefix('central_')
    mean_data = pd.read_csv("shap_analysis/shap_mean_data.csv").add_prefix('mean_')
    cna_data = pd.read_csv("shap_analysis/shap_cna_data.csv").add_prefix('cna_')
    mutations_data = pd.read_csv("shap_analysis/shap_mutations_data.csv").add_prefix('mutation_')
    central_values = pd.read_csv("shap_analysis/shap_central_values.csv").add_prefix('central_')
    mean_values = pd.read_csv("shap_analysis/shap_mean_values.csv").add_prefix('mean_')
    cna_values = pd.read_csv("shap_analysis/shap_cna_values.csv").add_prefix('cna_')
    mutations_values = pd.read_csv("shap_analysis/shap_mutations_values.csv").add_prefix('mutation_')

    top_mutations_list = mutations_values.abs().mean().sort_values(ascending=False).head(2).index.tolist()
    mutations_values = mutations_values.loc[:, mutations_values.columns.intersection(top_mutations_list)]
    # Scale all columns in mutations_values to the range (-0.05, 0.05) in one line
    mutations_values["mutation_TP53"] = scale_column(mutations_values["mutation_TP53"], -0.049, 0.049)
    mutations_values["mutation_AR"] = scale_column(mutations_values["mutation_AR"], -0.048, 0.048)

    all_values = pd.concat([central_values, mean_values, cna_values, mutations_values], axis=0)

    # Top n features to select
    n = 25

    top_n_features = all_values.abs().mean().sort_values(ascending=False).head(n)
    print(top_n_features)

    top_n_feature_names = top_n_features.index.tolist()
    filtered_all_values = pd.concat(
        [central_values.loc[:, central_values.columns.intersection(top_n_feature_names)],
         mean_values.loc[:, mean_values.columns.intersection(top_n_feature_names)],
         cna_values.loc[:, cna_values.columns.intersection(top_n_feature_names)],
         mutations_values.loc[:, mutations_values.columns.intersection(top_n_feature_names)]],
        axis=0,
        ignore_index=True
    )

    filtered_all_data = pd.concat(
        [central_data.loc[:, central_data.columns.intersection(top_n_feature_names)],
         mean_data.loc[:, mean_data.columns.intersection(top_n_feature_names)],
         cna_data.loc[:, cna_data.columns.intersection(top_n_feature_names)]],
        axis=0,
        ignore_index=True
    )

    filtered_all_values = filtered_all_values.reindex(columns=top_n_feature_names)
    filtered_all_data = filtered_all_data.reindex(columns=top_n_feature_names)

    # Shift non-NaN values to the top for all columns
    filtered_all_values = filtered_all_values.apply(lambda col: col.dropna().reset_index(drop=True).reindex(range(len(col)), fill_value=float('nan')))
    filtered_all_data = filtered_all_data.apply(lambda col: col.dropna().reset_index(drop=True).reindex(range(len(col)), fill_value=float('nan')))

    scaler = StandardScaler()
    standardized_filtered_all_data = pd.DataFrame(scaler.fit_transform(filtered_all_data), columns=filtered_all_data.columns)
    normalized_filtered_all_data = ((filtered_all_data - filtered_all_data.mean()) / filtered_all_data.std())
    edited_normalized_filtered_all_data = pd.concat([mutations_data.loc[:, mutations_data.columns.intersection(top_n_feature_names)], normalized_filtered_all_data], axis=1)
    edited_normalized_filtered_all_data.to_csv("shap_analysis/data.csv", index=False)

    return filtered_all_values, edited_normalized_filtered_all_data


def calculate_sensitivity_intervals(fpr, tpr):
    # Desired FPR levels for confidence interval calculation
    fpr_levels = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    # Interpolate TPR values at the given FPR levels
    interpolated_tpr = np.interp(fpr_levels, fpr, tpr)

    # Function to calculate confidence intervals for proportions (using normal approximation)
    def calculate_confidence_interval(p, n, alpha=0.05):
        z = norm.ppf(1 - alpha / 2)  # Z-score for the desired confidence level (95% CI)
        lower_bound = p - z * np.sqrt(p * (1 - p) / n)
        upper_bound = p + z * np.sqrt(p * (1 - p) / n)
        return max(0, lower_bound), min(1, upper_bound)

    # Assuming a hypothetical sample size (n) for confidence interval calculations
    # Replace `n` with the actual number of samples if known
    n = 48

    # Calculate confidence intervals for the interpolated TPR values
    ci_results = [
        {
            "FPR": fpr,
            "TPR (95% CI)": f"{tpr:.2f} ({calculate_confidence_interval(tpr, n)[0]:.2f} - {calculate_confidence_interval(tpr, n)[1]:.2f})"
        }
        for fpr, tpr in zip(fpr_levels, interpolated_tpr)
    ]

    # Convert results to a DataFrame for better visualization
    pd.DataFrame(ci_results).to_csv('ci_results.csv', index=False) 


def calculate_concordance(y_true, y_probs):
    pos_probs = y_probs[y_true == 1]
    neg_probs = y_probs[y_true == 0]

    total_pairs = len(pos_probs) * len(neg_probs)
    if total_pairs == 0:
        return None  # Avoid divide-by-zero

    concordant = 0
    ties = 0

    for pos_p in pos_probs:
        for neg_p in neg_probs:
            if pos_p > neg_p:
                concordant += 1
            elif pos_p == neg_p:
                ties += 1

    concordance_index = (concordant + 0.5 * ties) / total_pairs
    return concordance_index


def plot_validation_metrics_helper(data, metric, labels, output_path):
    """
    Helper function to plot a single metric as a boxplot.

    Args:
        data (dict): Dictionary containing metric data.
        metric (str): Metric to plot.
        labels (list): List of labels for the x-axis.
        output_path (str): Path to save the output plot.
    """
    fig, ax = plt.subplots()
    ax.boxplot(data[metric], patch_artist=True, boxprops=dict(facecolor='white', edgecolor='black'))
    ax.set_title(metric.capitalize())
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1)  # Assuming metric values are between 0 and 1
    plt.tight_layout()
    plt.savefig(output_path.replace(".png", f"_{metric}.png"), dpi=300)
    plt.close()


def plot_validation_metrics(csv_files, output_path):
    """
    Plot precision, f1_score, and concordance for all provided CSV files.

    Args:
        csv_files (list): List of file paths to CSV files.
        output_path (str): Path to save the output plots.
    """
    metrics = ['precision', 'f1_score', 'concordance']
    labels = ['ensemble', 'central', 'mean', 'cna']  # Renamed labels
    data = {metric: [] for metric in metrics}

    # Load data from all CSV files
    for file in csv_files:
        df = pd.read_csv(file)
        for metric in metrics:
            data[metric].append(df[metric].dropna().values)  # Collect data for each metric

    # Plot each metric using the helper function
    for metric in metrics:
        plot_validation_metrics_helper(data, metric, labels, output_path)


# List of CSV files to process
csv_files = [
    "validation_metrics/bar_validation_results_dctx_cbz_ensemble.png.csv",
    "validation_metrics/bar_validation_results_dctx_cbz_central_RNR_XGBClassifier.png.csv",
    "validation_metrics/bar_validation_results_dctx_cbz_mean_RNR_XGBClassifier.png.csv",
    "validation_metrics/bar_validation_results_dctx_cbz_cna_RNR_RandomForestClassifier.png.csv"
]

# Call the function to plot metrics
plot_validation_metrics(csv_files, "validation_metrics/combined_metrics_plot.png")
