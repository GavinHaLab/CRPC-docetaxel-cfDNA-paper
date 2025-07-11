import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu
from pingouin import ancova
from scipy import stats
from scipy.stats import chi2_contingency
from statsmodels.stats.multitest import fdrcorrection, multipletests
import statsmodels.api as sm
import seaborn as sns


# Function to find if a TFBS lies within a window in the CNV data
def find_cnv_for_tfbs(tfbs_row, cna_df):
    chr_match = cna_df['chr'] == tfbs_row['Chrom']
    start_in_range = cna_df['start'] <= tfbs_row['position']
    end_in_range = cna_df['end'] >= tfbs_row['position']
    
    # Find matching CNV rows
    matching_rows = cna_df[chr_match & start_in_range & end_in_range]
    
    # If no match, return the original TFBS row with NaN for CNV columns
    if matching_rows.empty:
        return pd.DataFrame([tfbs_row])
    else:
        # Add the TFBS information to each matching row
        matching_with_tfbs = matching_rows.copy()
        for col in tfbs_row.index:
            matching_with_tfbs[col] = tfbs_row[col]
        return matching_with_tfbs

def merge_df(df1, df2, output_file_name):
    merged_df = pd.merge(df1, df2)
    merged_df.to_csv(output_file_name, index=False)


# merge_df(pd.read_csv('all_dctx_cbz_cna.csv'), pd.read_csv('responder_list_cbz_only.csv'), 'cbz_only_cna.csv')

# # Load the ichorCNA copy number alteration data for ALL samples
# cna_data = pd.read_csv('cbz_only_cna.csv')
# # Load the transcription factor binding site (TFBS) data
# tfbs_data = pd.read_csv('/fh/fast/ha_g/user/adoebley/projects/griffin_revisions_1/sites/TFBS/10000_unfiltered_sites_CIS_BP_v2/AR.hg38.10000.txt', delimiter='\t')
# # Apply the function to each row in the TFBS data and concatenate the results
# final_df = pd.concat(tfbs_data.apply(lambda row: find_cnv_for_tfbs(row, cna_data), axis=1).tolist(), ignore_index=True)
# # Save the final output
# final_df.to_csv('dctx_tfbs_with_cna.csv', index=False)


def define_cytoband_with_samples():
    dctx_only_cna_df = pd.read_csv('cbz_only_cna.csv')
    ucsc_cytoband_df = pd.read_csv('ucsc_cytoband.csv')

    # Update the mapping logic to ensure chromosome numbers also match
    cytoband_intervals = pd.IntervalIndex.from_arrays(ucsc_cytoband_df['chromStart'], ucsc_cytoband_df['chromEnd'], closed='both')

    # Map samples to band names with chromosome matching and interval checking
    dctx_only_cna_df['bandName'] = dctx_only_cna_df.apply(
        lambda row: ucsc_cytoband_df.loc[
            (ucsc_cytoband_df['chrom'] == row['chr']) &
            cytoband_intervals.contains(row['start']) &
            cytoband_intervals.contains(row['end']), 'bandName'
        ].values[0] if any(
            (ucsc_cytoband_df['chrom'] == row['chr']) &
            cytoband_intervals.contains(row['start']) &
            cytoband_intervals.contains(row['end'])
        ) else None,
        axis=1
    )

    dctx_only_cna_df.to_csv("cbz_only_cna_bandlevel.csv", index=False)


def run_mannwhitney(df, group_by, gain_flag, output_file_name):
    df = df.dropna(subset=[group_by])
    df = df[df['response'].isin(['Y', 'N'])]

    results = []

    # Loop through each unique chromosome band
    for band in df['chrBandName'].unique():
        # Filter the data for the current band
        band_data = df[df['chrBandName'] == band]
        
        if gain_flag:
            # if not ((band_data['response'] == 'N') & (band_data['relative_copy_number'] >= 1.5)).any() or not ((band_data['response'] == 'Y') & (band_data['relative_copy_number'] >= 1.5)).any():
            #     continue
            # Separate the data by response groups with gains
            group1 = band_data[(band_data['response'] == 'N') & (band_data['relative_copy_number'] >= 1.5)][group_by]
            group2 = band_data[(band_data['response'] == 'Y') & (band_data['relative_copy_number'] >= 1.5)][group_by]
        else:
            # if not ((band_data['response'] == 'N') & (band_data['relative_copy_number'] <= 0.5)).any() or not ((band_data['response'] == 'Y') & (band_data['relative_copy_number'] <= 0.5)).any():
            #     continue
            # Separate the data by response groups with deletions
            group1 = band_data[(band_data['response'] == 'N') & (band_data['relative_copy_number'] <= 0.99)][group_by]
            group2 = band_data[(band_data['response'] == 'Y') & (band_data['relative_copy_number'] <= 0.99)][group_by]
        
        # Ensure both groups have data before performing the test
        if len(group1) > 0 and len(group2) > 0:
            # Perform the Mann-Whitney U test
            u_stat, p_value = mannwhitneyu(group1, group2, alternative='two-sided')
            
            # Calculate log fold change (log2 of the ratio of means)
            mean_group1 = np.mean(group1)
            mean_group2 = np.mean(group2)
            log_fold_change = np.log2(mean_group2 / mean_group1) if mean_group1 > 0 and mean_group2 > 0 else np.nan
            
            # Store the results
            results.append({
                'chrBandName': band,
                'Mann-Whitney U Statistic': u_stat,
                'p-value': p_value,
                'Log Fold Change (Responder / Non-Responder copy number)': log_fold_change
            })

    results_df = pd.DataFrame(results)
    # Apply False Discovery Rate (FDR) correction to the p-values
    if not results_df.empty:
        p_values = results_df['p-value'].values
        _, corrected_p_values, _, _ = multipletests(p_values, method='fdr_bh')
        results_df['FDR-corrected p-value'] = corrected_p_values
    else:
        results_df['FDR-corrected p-value'] = []

    results_df.to_csv(output_file_name, index=False)



# run_mannwhitney(pd.read_csv("dctx_only_cna_bandlevel.csv"), 'logR_Copy_Number', "dctx_cna_mannwhitney_(logR_Copy_Number)_results.csv")
# run_mannwhitney(pd.read_csv("cna_analysis/new_band_cn_fet.csv"), 'relative_copy_number', True, "dctx_cna_mannwhitney_(relative_copy_number)_results_gain.csv")
run_mannwhitney(pd.read_csv("cna_analysis/new_band_cn_fet.csv"), 'relative_copy_number', False, "dctx_cna_mannwhitney_(relative_copy_number)_results_deletion.csv")

# dfdc =  pd.read_csv("dctx_only_cna_bandlevel.csv")
# unique_bands = dfdc['chrBandName'].unique()
# result = pd.DataFrame(unique_bands, columns=["test"])
# result.to_csv("result.csv", index=False)
