import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu
from pingouin import ancova
from scipy import stats
from scipy.stats import chi2_contingency, fisher_exact
from statsmodels.stats.multitest import fdrcorrection, multipletests
import statsmodels.api as sm
import seaborn as sns
from matplotlib.patches import Patch

# Set font to Helvetica
plt.rcParams['font.family'] = 'Helvetica'

def plot_cna_proportion_arms():
    data = pd.read_csv('new_arms_cn_fet_counts.csv')
    
    # Calculate proportions for each category
    data["NR_total"] = data["NR_deletion"] + data["NR_neutral"] + data["NR_gain"]
    data["R_total"] = data["R_deletion"] + data["R_neutral"] + data["R_gain"]
    
    data["NR_deletion"] = data["NR_deletion"] / data["NR_total"]
    data["NR_neutral"] = data["NR_neutral"] / data["NR_total"]
    data["NR_gain"] = data["NR_gain"] / data["NR_total"]
    data["R_deletion"] = data["R_deletion"] / data["R_total"]
    data["R_neutral"] = data["R_neutral"] / data["R_total"]
    data["R_gain"] = data["R_gain"] / data["R_total"]

    # Adjust chromosome order to match dataset
    chromosome_order = [
        "chr1p", "chr1q", "chr2p", "chr2q", "chr3p", "chr3q", "chr4p", "chr4q", "chr5p", "chr5q",
        "chr6p", "chr6q", "chr7p", "chr7q", "chr8p", "chr8q", "chr9p", "chr9q", "chr10p", "chr10q",
        "chr11p", "chr11q", "chr12p", "chr12q", "chr13p", "chr13q", "chr14p", "chr14q", "chr15p", "chr15q",
        "chr16p", "chr16q", "chr17p", "chr17q", "chr18p", "chr18q", "chr19p", "chr19q", "chr20p", "chr20q",
        "chr21p", "chr21q", "chr22p", "chr22q", "chrXp", "chrXq"
    ]

    # Ensure chromosome arm order
    data["arm"] = pd.Categorical(data["arm"], categories=chromosome_order, ordered=True)
    data = data.sort_values("arm")

    # Define colors
    colors = {"deletion": "#1C75BC", "neutral": "#939598", "gain": "#F7941D"}
    bar_width = 0.35

    # Plot
    plt.figure(figsize=(16, 4))
    for i, response in enumerate(["NR", "R"]):
        bar_positions = np.arange(len(data)) + (i * bar_width)
        bar_bottom = np.zeros(len(data))
        
        for category in ["deletion", "neutral", "gain"]:
            plt.bar(
                bar_positions,
                data[f"{response}_{category}"].values,
                bottom=bar_bottom,
                width=bar_width,
                color=colors[category],
                edgecolor='black',
                linewidth=1,
                label=f"{category}" if i == 0 else ""
            )
            bar_bottom += data[f"{response}_{category}"].values

    # Customize the plot
    plt.xticks(np.arange(len(data["arm"])) + bar_width / 2,
               data["arm"], rotation=90)
    plt.xlabel("Chromosome Arm")
    plt.ylabel("Proportion")
    plt.title("Proportion of CNA Types by Chromosome Arm (Responders vs. Non-Responders)")
    plt.ylim(0, 1)
    plt.legend(title="CNA Category", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('new_chr_arm_plot.png', dpi=300)


def plot_cna_proportion_bands(data, band_list, output_name):
    
    # Ensure the bands are plotted in the order of band_list
    data = data[data['chrBandName'].isin(band_list)]
    data["chrBandName"] = pd.Categorical(data["chrBandName"], categories=band_list, ordered=True)
    data = data.sort_values("chrBandName") # type: ignore
    
    data["NR_total"] = data["NR_deletion"] + data["NR_neutral"] + data["NR_gain"]
    data["R_total"] = data["R_deletion"] + data["R_neutral"] + data["R_gain"]
    
    data["NR_deletion"] = data["NR_deletion"] / data["NR_total"]
    data["NR_neutral"] = data["NR_neutral"] / data["NR_total"]
    data["NR_gain"] = data["NR_gain"] / data["NR_total"]
    data["R_deletion"] = data["R_deletion"] / data["R_total"]
    data["R_neutral"] = data["R_neutral"] / data["R_total"]
    data["R_gain"] = data["R_gain"] / data["R_total"]

    # Define colors
    colors = {"deletion": "#1C75BC", "neutral": "#939598", "gain": "#F7941D"}
    bar_width = 0.35

    # Plot
    plt.figure(figsize=(18, 5))
    for i, response in enumerate(["NR", "R"]):
        bar_positions = np.arange(len(data)) + (i * bar_width)
        bar_bottom = np.zeros(len(data))
        
        for category in ["deletion", "neutral", "gain"]:
            plt.bar(
                bar_positions,
                data[f"{response}_{category}"].values,
                bottom=bar_bottom,
                width=bar_width,
                color=colors[category],
                edgecolor='black',
                linewidth=1,
                label=f"{category}" if i == 0 else ""
            )
            bar_bottom += data[f"{response}_{category}"].values

    # Customize the plot
    plt.xticks(np.arange(len(data["chrBandName"])) + bar_width / 2, data["chrBandName"], rotation=90)
    plt.xlabel("Chromosome Band")
    plt.ylabel("Proportion")
    plt.title("Proportion of CNA Types by Chromosome Band (Responders vs. Non-Responders)")
    plt.ylim(0, 1)
    plt.legend(title="CNA Category", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(f'{output_name}_chr_bands_plot.png', dpi=300)


def plot_chi_square_boxplot(file_path, output_path):
    df = pd.read_csv(file_path)
    df = df.replace(0, 0.0000001)  # Avoid zeros for log scales or stats

    # Calculate proportions for each arm/band
    df["NR_total"] = df["NR_deletion"] + df["NR_neutral"] + df["NR_gain"]
    df["R_total"] = df["R_deletion"] + df["R_neutral"] + df["R_gain"]

    df["NR_deletion_prop"] = df["NR_deletion"] / df["NR_total"]
    df["NR_neutral_prop"] = df["NR_neutral"] / df["NR_total"]
    df["NR_gain_prop"] = df["NR_gain"] / df["NR_total"]
    df["R_deletion_prop"] = df["R_deletion"] / df["R_total"]
    df["R_neutral_prop"] = df["R_neutral"] / df["R_total"]
    df["R_gain_prop"] = df["R_gain"] / df["R_total"]

    # Prepare data for boxplot: melt into long format
    plot_data = []
    for category in ["deletion", "neutral", "gain"]:
        for response in ["NR", "R"]:
            values = df[f"{response}_{category}_prop"].values
            plot_data.extend([
                {"Category": category, "Response": response, "Proportion": v}
                for v in values
            ])

    plot_df = pd.DataFrame(plot_data)

    # Set order so NR and R for each category are adjacent
    order = []
    palette = {}
    for cat in ["deletion", "neutral", "gain"]:
        for resp, color in zip(["NR", "R"], ["C4", "C8"]):
            order.append(f"{cat}_{resp}")
            palette[f"{cat}_{resp}"] = color

    plot_df["xcat"] = plot_df["Category"] + "_" + plot_df["Response"]

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.boxplot(
        x="xcat",
        y="Proportion",
        data=plot_df,
        order=order,
        palette=palette,
        width=0.6,
        showfliers=False,
        ax=ax
    )
    # # Randomly sample up to 50 points per group for stripplot
    # sampled_plot_df = (
    #     plot_df.groupby("xcat", group_keys=False)
    #     .apply(lambda g: g.sample(n=min(len(g), 25), random_state=42))
    #     .reset_index(drop=True)
    # )

    # sns.stripplot(
    #     x="xcat",
    #     y="Proportion",
    #     data=sampled_plot_df,
    #     order=order,
    #     color="black",
    #     dodge=False,
    #     jitter=True,
    #     size=2,
    #     alpha=0.7,
    #     ax=ax
    # )

    # Set x-ticks to show category and response, grouped
    ax.set_xticklabels([f"{cat}\n{resp}" for cat, resp in [item.split('_') for item in order]])
    plt.ylim(0, 1.0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()



def plot_chi_square_bar_graph(file_path, output_path):
    df = pd.read_csv(file_path)
    df = df.replace(0, 0.0000001)  # Replace all 0 cells with 0.0000001

    # Sum across all chromosome arms/bands for each category
    NR_deletion = df["NR_deletion"]
    R_deletion = df["R_deletion"]
    NR_neutral = df["NR_neutral"]
    R_neutral = df["R_neutral"]
    NR_gain = df["NR_gain"]
    R_gain = df["R_gain"]

    # Create separate 2x2 contingency tables for each comparison
    contingency_deletions = [NR_deletion.to_list(), R_deletion.to_list()]
    contingency_neutral = [NR_neutral.to_list(), R_neutral.to_list()]
    contingency_gain = [NR_gain.to_list(), R_gain.to_list()]

    # Perform chi-square tests
    chi2_del, p_del, _, _ = stats.chi2_contingency(contingency_deletions)
    chi2_neu, p_neu, _, _ = stats.chi2_contingency(contingency_neutral)
    chi2_gain, p_gain, _, _ = stats.chi2_contingency(contingency_gain)

    # Print results
    print(f"P-value for Deletions (NR vs R): {p_del}")
    print(f"P-value for Neutral (NR vs R): {p_neu}")
    print(f"P-value for Gains (NR vs R): {p_gain}")

    # Data for plotting
    NR_total_count = NR_deletion.sum() + NR_neutral.sum() + NR_gain.sum()
    R_total_count = R_deletion.sum() + R_neutral.sum() + R_gain.sum()
    
    NR_counts_proportion = [NR_deletion.sum() / NR_total_count, NR_neutral.sum() / NR_total_count, NR_gain.sum() / NR_total_count]
    R_counts_proportion = [R_deletion.sum() / R_total_count, R_neutral.sum() / R_total_count, R_gain.sum() / R_total_count]

    # Compute standard deviations (assuming Poisson distribution: sqrt(count))
    NR_std_error = [np.sqrt(p * (1 - p) / NR_total_count) for p in NR_counts_proportion]
    R_std_error = [np.sqrt(p * (1 - p) / R_total_count) for p in R_counts_proportion]

    x = np.arange(3)  # X locations for the bars
    bar_width = 0.2

    # Create plot
    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot bars with error bars
    ax.bar(x - 0.15, NR_counts_proportion, yerr=NR_std_error, width=bar_width, label="Non-Responder", capsize=5, facecolor='white', edgecolor='black')
    ax.bar(x + 0.15, R_counts_proportion, yerr=R_std_error, width=bar_width, label="Responder", capsize=5, facecolor='black', edgecolor='black')
    
    # ax.axes.xaxis.set_ticklabels([])
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.yaxis.set_ticklabels([])
    ax.set_axisbelow(True)
    
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)


def calculate_gene_proportions(values):
    neutral_count = 0.0
    deletion_count = 0.0
    gain_count = 0.0
    for value in values:
        if value <= 0.5:
            deletion_count += 1
        elif value >= 1.5:
            gain_count += 1
        else:
            neutral_count += 1
    total_count = deletion_count + neutral_count + gain_count
    return deletion_count / total_count, neutral_count / total_count, gain_count / total_count


def plot_cna_genes(output_path):
    ensembl_df = pd.read_csv('gene_level/ensembl_gene_annotations.csv')
    gene_counts_df = pd.read_csv('gene_level/merged_gene_counts.csv')
    
    # Create a dictionary for quick gene-to-chromosome and type lookup
    gene_to_chr = ensembl_df.set_index("Gene").apply(lambda row: {"Chr": row["Chr"], "Type": row["Type"]}, axis=1).to_dict()
    
    R_counts_df = gene_counts_df[gene_counts_df["response"] == "Y"]
    NR_counts_df = gene_counts_df[gene_counts_df["response"] == "N"]
    
    num_genes = len(R_counts_df.columns) - 2  # Exclude 'response' and 'sample' columns
    max_genes_to_plot = 3000  # Limit the number of genes to plot
    if num_genes > max_genes_to_plot:
        print(f"Too many genes ({num_genes}). Subsampling to {max_genes_to_plot} genes.")
        sampled_genes = np.random.choice(
            [col for col in R_counts_df.columns if col not in ["response", "sample"]],
            size=max_genes_to_plot,
            replace=False
        )
        R_counts_df = R_counts_df[sampled_genes]
        num_genes = max_genes_to_plot

    # Precompute proportions for all genes
    gene_proportions = []
    for name, values in R_counts_df.items():
        try:
            del_prop, _, gain_prop = calculate_gene_proportions(values)
        except (ValueError, TypeError) as e:
            print(f"Error processing values: {values}. Skipping. Error: {e}")
            continue
        gene_info = gene_to_chr.get(name, None)
        if gene_info is None:
            print(f"Warning: Gene '{name}' not found in ensembl annotations. Skipping.")
            continue
        chr = gene_info["Chr"]
        gene_type = gene_info["Type"]
        if gene_type != "protein_coding":
            continue
        # Convert chromosome to integer for sorting
        if chr == "X":
            chr = 23
        elif chr == "Y":
            chr = 24
        else:
            try:
                chr = int(chr)
            except ValueError:
                print(f"Warning: Invalid chromosome '{chr}' for gene '{name}'. Skipping.")
                continue
        gene_proportions.append((chr, del_prop, gain_prop, name, gene_type))

    # Sort by chromosome and assign sequential x-positions
    gene_proportions.sort(key=lambda x: x[0])  # Sort by chromosome as integers
    x_positions = list(range(len(gene_proportions)))  # Sequential x-positions

    print(f"Plotting {len(gene_proportions)} genes.")
    num_genes = len(gene_proportions)
    bar_width = 0.8 / num_genes  # Adjust bar width based on the number of genes
    fig_width = min(max(8, num_genes * 0.02), 200)  # Scale figure width dynamically with an upper limit
    fig, ax = plt.subplots(figsize=(fig_width, 6))

    # Batch plotting
    for x, (chr, del_prop, gain_prop, name, gene_type) in zip(x_positions, gene_proportions):
        ax.bar(x, -del_prop, width=bar_width, facecolor='C0', edgecolor='C0')  # Deletions
        ax.bar(x, gain_prop, width=bar_width, facecolor='C1', edgecolor='C1')  # Gains

    # Customize x-axis to show chromosome labels
    chr_labels = [f"chr{int(chr)}" if chr != 23 else "chrX" for chr, *_ in gene_proportions]
    ax.set_xticks([x_positions[i] for i in range(len(x_positions)) if i == 0 or gene_proportions[i][0] != gene_proportions[i - 1][0]])
    ax.set_xticklabels([chr_labels[i] for i in range(len(chr_labels)) if i == 0 or gene_proportions[i][0] != gene_proportions[i - 1][0]], rotation=90)

    plt.ylim(-1.0, 1.0)
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)


def collate_cna_ichor():
    all_seg_list = []
    directory = '/fh/fast/ha_g/user/azimmer/ichorCNA_Docetexal/scripts/ichorCNA_n_p_rerun/results/ichorCNA'
    
    for filename in os.listdir(directory):
        if filename == ".DS_Store":
            continue
        
        new_dir = os.path.join(directory, filename)
        seg_path = os.path.join(new_dir, f"{filename}.cna.seg")

        seg_df = pd.read_csv(seg_path, sep='\t')
        seg_df.columns = seg_df.columns.str.replace(r'^[^\.]+\.', '', regex=True)
        seg_df.insert(0, 'sample', filename)

        all_seg_list.append(seg_df)
    
    seg_df = pd.concat(all_seg_list, ignore_index=True)
    responder_df = pd.read_csv('responder_list_dctx_only.csv')
    seg_merged = pd.merge(responder_df, seg_df, on='sample', how='left')
    seg_merged.to_csv('gene_level/merged_gene_counts.csv', index=False)
    seg_merged.to_csv("collated_ichor_cna_dctx_results.csv", index=False, header=True)


def load_ucsc_arms(cytoband_path):
    """
    Parses UCSC cytoBand.txt file and returns a dict of arm boundaries.
    Output: { 'chr1p': (start, end), 'chr1q': (start, end), ... }
    """
    cyto = pd.read_csv(cytoband_path, sep='\t', header=None,
                       names=['chr', 'start', 'end', 'band', 'stain'])
    
    arms = {}
    for chrom in cyto['chr'].unique():
        chr_data = cyto[cyto['chr'] == chrom]
        acen_bands = chr_data[chr_data['stain'] == 'acen']
        if acen_bands.empty or len(acen_bands) != 2:
            continue  # Skip if centromere data is missing or malformed

        # p arm: from start to acen start
        p_start = chr_data['start'].astype(int).min()
        p_end = acen_bands['start'].astype(int).min()
        arms[f"{chrom.replace('chr', '')}p"] = (p_start, p_end)

        # q arm: from acen end to chromosome end
        q_start = acen_bands['end'].astype(int).max()
        q_end = chr_data['end'].astype(int).max()  # Ensure 'end' is treated as integers
        arms[f"{chrom.replace('chr', '')}q"] = (q_start, q_end)
    print(arms)
    
    return arms


def assign_chromosome_arm(row, arm_boundaries):
    chrom = row['chr'].replace('chr', '')
    for arm, (start, end) in arm_boundaries.items():
        if not arm.startswith(chrom):
            continue
        if row['start'] >= int(start) and row['end'] <= int(end):
            return arm
        elif row['start'] < int(end) and row['end'] > int(start):
            return 'split'  # Crosses the boundary
    return None


def extract_avg_copy_number_by_ucsc_arm():
    """
    Uses UCSC cytoband to assign chromosome arms and calculate average copy number per arm.
    """
    arm_boundaries = load_ucsc_arms("/fh/fast/ha_g/user/dchen4/griffin_analysis/ucsc_cytoband.txt")
    
    df = pd.read_csv("collated_ichor_cna_dctx_results.csv")
    df['arm'] = df.apply(assign_chromosome_arm, axis=1, arm_boundaries=arm_boundaries)

    # Split segments that cross arms
    split_segments = []
    for _, row in df[df['arm'] == 'split'].iterrows():
        chrom = row['chr'].replace('chr', '')
        for arm in ['p', 'q']:
            arm_key = f"{chrom}{arm}"
            if arm_key not in arm_boundaries:
                continue
            a_start, a_end = arm_boundaries[arm_key]
            overlap_start = max(row['start'], int(a_start))
            overlap_end = min(row['end'], int(a_end))
            if overlap_start < overlap_end:
                split_segments.append({
                    'response': row['response'],
                    'sample': row['sample'],
                    'chr': row['chr'],
                    'start': overlap_start,
                    'end': overlap_end,
                    'copy.number': row['copy.number'],
                    'arm': arm_key
                })

    df = df[df['arm'] != 'split']
    if split_segments:
        df = pd.concat([df, pd.DataFrame(split_segments)], ignore_index=True)

    df['length'] = df['end'] - df['start']

    result = df.groupby(['response', 'sample', 'arm']).apply(
        lambda g: pd.Series({
            'relative_copy_number': ((g['copy.number'] * g['length']).sum() / g['length'].sum()) / 2
        })
    ).reset_index()

    result.to_csv('new_arms_cn_fet.csv', index=False)


def categorize_copy_number(val):
    if val <= 0.5:
        return "<=0.5"
    elif val >= 1.5:
        return ">=1.5"
    else:
        return "=1"


def create_arm_fet_counts():
    """
    Converts per-sample relative copy number data into a count summary table.
    """
    df = pd.read_csv("new_arms_cn_fet.csv")

    # Bin into CNA categories
    df['category'] = df['relative_copy_number'].apply(categorize_copy_number)

    # Filter only the desired bins
    df = df[df['category'].isin(["<=0.5", "=1", ">=1.5"])]

    # Normalize arm names to have 'chr' prefix
    df['arm'] = df['arm'].apply(lambda x: f"chr{x}" if not x.startswith("chr") else x)

    # Pivot into summary counts
    # Ensure df is a DataFrame
    if isinstance(df, pd.Series):
        df = df.to_frame()

    pivot = pd.pivot_table(
        df,
        index='arm',
        columns=['response', 'category'],
        aggfunc='size',
        fill_value=0
    )

    # Flatten multi-index columns
    pivot.columns = [f"{'NR' if r == 'N' else 'R'}{c}" for r, c in pivot.columns]

    # Ensure all expected columns are present
    for col in ['NR<=0.5', 'NR=1', 'NR>=1.5', 'R<=0.5', 'R=1', 'R>=1.5']:
        if col not in pivot.columns:
            pivot[col] = 0

    # Reorder columns
    ordered_cols = ['NR<=0.5', 'NR=1', 'NR>=1.5', 'R<=0.5', 'R=1', 'R>=1.5']
    pivot = pivot[ordered_cols].reset_index()

    pivot.to_csv("new_arms_cn_fet_counts.csv", index=False)


def chr_arm_fishers():
    # Initialize lists to store p-values
    df = pd.read_csv("new_arms_cn_fet_counts.csv")
    deletion_p_values = []
    gain_p_values = []

    # Loop through each row to perform Fisher's exact test
    for _, row in df.iterrows():
        # Deletion contingency table
        deletion_table = [
            [row['NR_deletion'], row['NR_neutral'] + row['NR_gain']],
            [row['R_deletion'], row['R_neutral'] + row['R_gain']]
        ]
        _, p_deletion = fisher_exact(deletion_table)
        
        # Gain contingency table
        gain_table = [
            [row['NR_gain'], row['NR_neutral'] + row['NR_deletion']],
            [row['R_gain'], row['R_neutral'] + row['R_deletion']]
        ]
        _, p_gain = fisher_exact(gain_table)
        
        deletion_p_values.append(p_deletion)
        gain_p_values.append(p_gain)

    # Add the p-values to the dataframe
    df['deletion_p_value'] = deletion_p_values
    df['gain_p_value'] = gain_p_values

    print(df)
    

def match_cytobands_with_sample(cytoband_df, sample_df):
    results = []

    for _, band in cytoband_df.iterrows():
        chrom = band['chrom']
        band_start = band['chromStart']
        band_end = band['chromEnd']
        chrombandName = band['chrombandName']

        # Filter matching rows in the sample table by overlapping coordinates and chromosome
        overlapping = sample_df[
            (sample_df['chr'] == chrom) &
            (sample_df['end'] > band_start) &
            (sample_df['start'] < band_end)
        ]

        if not overlapping.empty:
            # Group by sample and response, then average copy number for each group
            grouped = overlapping.groupby(['sample', 'response'])['copy.number'].mean().reset_index()
            for _, row in grouped.iterrows():
                results.append({
                    'chrombandName': chrombandName,
                    'sample': row['sample'],
                    'response': row['response'],
                    'avg_copy_number': row['copy.number']
                })

    pd.DataFrame(results).to_csv('new_band_cn_fet.csv', index=False)


def create_band_fet_counts():
    """
    Converts per-sample relative copy number data into a count summary table.
    """
    df = pd.read_csv("new_band_cn_fet.csv")

    # Bin into CNA categories
    df['category'] = df['relative_copy_number'].apply(categorize_copy_number)

    # Filter only the desired bins
    df = df[df['category'].isin(["<=0.5", "=1", ">=1.5"])]

    # Normalize arm names to have 'chr' prefix
    df['chrBandName'] = df['chrBandName'].apply(lambda x: f"chr{x}" if not x.startswith("chr") else x)

    # Pivot into summary counts
    # Ensure df is a DataFrame
    if isinstance(df, pd.Series):
        df = df.to_frame()

    pivot = pd.pivot_table(
        df,
        index='chrBandName',
        columns=['response', 'category'],
        aggfunc='size',
        fill_value=0
    )

    # Flatten multi-index columns
    pivot.columns = [f"{'NR' if r == 'N' else 'R'}{c}" for r, c in pivot.columns]

    # Ensure all expected columns are present
    for col in ['NR<=0.5', 'NR=1', 'NR>=1.5', 'R<=0.5', 'R=1', 'R>=1.5']:
        if col not in pivot.columns:
            pivot[col] = 0

    # Reorder columns
    ordered_cols = ['NR<=0.5', 'NR=1', 'NR>=1.5', 'R<=0.5', 'R=1', 'R>=1.5']
    pivot = pivot[ordered_cols].reset_index()

    pivot.to_csv("new_band_cn_fet_counts.csv", index=False)


## -----------------------------------------------------------
## FOR CHROMOSOME ARM STUFF
# extract_avg_copy_number_by_ucsc_arm()
# create_arm_fet_counts()
# plot_cna_proportion_arms()
# chr_arm_fishers()

file_path = "new_arms_cn_fet_counts.csv"
output_path = "new_chr_arms_chi_square_boxplot.png"
plot_chi_square_boxplot(file_path, output_path)
## -----------------------------------------------------------


## -----------------------------------------------------------
## FOR CHROMOSOME BAND STUFF
# match_cytobands_with_sample(pd.read_csv("/fh/fast/ha_g/user/dchen4/griffin_analysis/ucsc_cytoband.csv"), pd.read_csv('collated_ichor_cna_dctx_results.csv'))
# create_band_fet_counts()

#significant band list defined by Mann-Whitney-U tests comparing significant gains/deletions between R vs NR
# total_significant_band_list = [
#     "chr11p15.4", "chr11q24.1", "chr11q24.2", "chr11p13", "chr11p15.2", "chr11p15.1", "chr11p12",
#     "chr11p11.2", "chr11p14.1", "chr10q11.22", "chr10q11.23", "chr10q21.1", "chr11p15.5", "chr11p15.3", "chr11q24.3", "chr11q25", "chr3p14.1",
#     "chr3p14.2", "chr11q23.2", "chr3p13", "chr3p21.2", "chr12q24.11", "chr11q13.1", "chr3p21.1",
#     "chr16q13", "chr3p14.3", "chr1p33", "chr1p36.31", "chr6q24.1",
#     "chr6q24.2", "chr2p25.2", "chr2p25.3", "chr2q21.2", "chr6q24.3", "chr1p34.1", "chr2q21.3", "chr1p36.13", 
#      "chr11q14.2", "chr11q14.3", "chr6q23.3",  "chr11q13.3",
# ]
# r_significant_band_list = [
#     "chr3q25.1", "chr3q25.2", "chr8q11.21", "chr1p36.12", "chr1p35.2", "chr1p35.1", "chr1p34.3",
#     "chr3q25.31", "chr13q31.2", "chr8p23.2", "chr10q21.1", "chr10q24.31", "chr10q24.32", "chr8q12.2",
#     "chr1p36.11", "chr1p35.3", "chr1p31.2", "chr5p13.2", "chr3q22.2", "chr8p23.3", "chr11p14.1",
#     "chr11q23.3", "chr1p21.3", "chr11q22.3", "chr2q14.3", "chr6q25.3", "chr11q23.2", "chr11p13",
#     "chr3p13", "chr1p31.3", "chr16q21", "chr16q22.1", "chr10q21.3", "chr2q21.3", "chr6q23.3",
#     "chr5q31.3", "chr8p22", "chr12q21.31", "chr1p32.3", "chr1p36.21"
# ]

# nr_significant_band_list = [
#     "chr2p25.2", "chr2p25.3", "chr2p25.1", "chr2q35", "chrXp22.33", "chrXp22.32", "chrXp22.31",
#     "chrXp22.2", "chrXp22.13", "chr7q31.31", "chrXp22.12", "chrXp22.11", "chrXp21.3", "chrXp21.2",
#     "chr7q31.2", "chr2p24.3", "chr2p24.2", "chr2p24.1", "chr2p23.3", "chr11q13.4", "chr5q23.1",
#     "chr13q12.3", "chr18q11.2", "chr5q21.3", "chr4q31.1", "chr10p12.2", "chrXq27.3", "chr6q13",
#     "chr13q12.2", "chrXq21.1", "chr13q13.3", "chr2q24.3", "chr9q21.13", "chr5q11.2", "chr2q14.1",
#     "chr2q11.2", "chr17q25.1", "chr4q27", "chr4q28.1", "chr6q15"
# ]
# total_data = pd.read_csv('new_band_cn_fet_counts.csv')
# plot_cna_proportion_bands(total_data, total_significant_band_list, "total")
# plot_cna_proportion_bands(total_data, r_significant_band_list, "r")
# plot_cna_proportion_bands(total_data, nr_significant_band_list, "nr")

file_path = "new_band_cn_fet_counts.csv"
output_path = "new_chr_bands_chi_square_boxplot.png"
plot_chi_square_boxplot(file_path, output_path)
## -----------------------------------------------------------


## -----------------------------------------------------------
## FOR CHROMOSOME GENE STUFF
# responder_df = pd.read_csv('responder_list_dctx_only.csv')
# gene_df = pd.read_csv('gene_level/cn_status_focal.csv')
# gene_count_merged = pd.merge(responder_df, gene_df, on='sample', how='left')
# gene_count_merged.to_csv('gene_level/merged_gene_counts.csv', index=False)


# resistance_genes_df = pd.read_csv('../../gsea_analysis/dctx_resistance_genes.csv')
# gain_genes_df = pd.read_csv('../../gsea_analysis/gain.txt', sep='\t')
# loss_genes_df = pd.read_csv('../../gsea_analysis/loss.txt', sep='\t')

# merged_df = pd.merge(resistance_genes_df, gain_genes_df, on='Gene', how='left')
# merged_df.to_csv('../../gsea_analysis/resistance_genes_with_gain.csv', index=False)
# merged_df = pd.merge(resistance_genes_df, loss_genes_df, on='Gene', how='left')
# merged_df.to_csv('../../gsea_analysis/resistance_genes_with_loss.csv', index=False)
## -----------------------------------------------------------



