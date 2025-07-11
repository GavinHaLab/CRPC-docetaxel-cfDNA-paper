import pandas as pd
import numpy as np
from scipy.stats import ttest_ind, mannwhitneyu, chi2_contingency

def extract_ataqseq_features():
    dctx_clinical_df = pd.read_csv('responder_list_dctx_only.csv')
    triton_results_df = pd.read_csv('/fh/fast/ha_g/user/dchen4/patient_ataqseq_data/ATACseq/TritonCompositeFM_3.tsv', delimiter='\t')

    central_df = triton_results_df[triton_results_df['site'] == 'DBA_Gain_sites3']
    result_central_depth_df = central_df.pivot(index='sample', columns='feature', values='value')
    merged_df1 = pd.merge(dctx_clinical_df, result_central_depth_df, on='sample', how='inner')
    merged_df1.to_csv('triton_DBA_Gain_sites3.csv', index=False)

    mean_df = triton_results_df[triton_results_df['site'] == 'DBA_loss_sites3']
    result_mean_depth_df = mean_df.pivot(index='sample', columns='feature', values='value')
    merged_df2 = pd.merge(dctx_clinical_df, result_mean_depth_df, on='sample', how='inner')
    merged_df2.to_csv('triton_DBA_loss_sites3.csv', index=False)


def read_npz_file():
    # Load the file
    data = np.load('../patient_ataqseq_data/Triton_Docetexal_ATACseq/results_2/GENP1147-1_TritonProfiles.npz')
    loss = data['DBA_loss_sites2']
    gain = data['DBA_Gain_sites2']
    
    pd.DataFrame(loss).to_csv(f"triton_loss_sites_testing.csv")
    pd.DataFrame(gain).to_csv(f"triton_gain_sites_testing.csv")

    # Close the file
    data.close()


def univariate_analysis(data):
    # Separate the response column
    target = 'response'
    features = [col for col in data.columns if col != target and col != 'sample']

    # Store results
    results = []

    # Perform analysis for numerical features
    for feature in features:
        group_Y = data[data[target] == 'Y'][feature].dropna()
        group_N = data[data[target] == 'N'][feature].dropna()

        # Perform Mann-Whitney U test for non-parametric comparison
        stat, p_value = mannwhitneyu(group_Y, group_N, alternative='two-sided')
        results.append({
            'Feature': feature,
            'Statistic': stat,
            'P-value': p_value
        })

    # Convert results to a DataFrame for better readability
    results_df = pd.DataFrame(results)
    
    results_df.to_csv(f"mannwhitneyu_analysis_loss_sites.csv", index=False)


# Load the data
# data = pd.read_csv('triton_DBA_loss_sites3.csv')
# univariate_analysis(data)\

read_npz_file()