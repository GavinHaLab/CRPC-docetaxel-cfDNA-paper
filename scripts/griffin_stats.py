import csv
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.stats import mannwhitneyu
from scipy.interpolate import make_interp_spline
from pingouin import ancova
from tabulate import tabulate
from scipy import stats
from statsmodels.stats.multitest import fdrcorrection
import statsmodels.api as sm
from sklearn.preprocessing import quantile_transform, StandardScaler

from bioinfokit import analys, visuz
from collate_results import tfx_correction
import seaborn as sns
from helper import label_responders

# global variables
TFx_threshold = 0.05
ploidy_threshold = 0
cov_threshold = 0.05


def create_dirs(dir):
    script_dir = os.path.dirname(__file__)
    results_dir = os.path.join(script_dir, f"plots/{dir}/")
    os.makedirs(results_dir, exist_ok=True)


def calculate_coverage(directory, output_path, col_name, site_names, agg_file_path):
    with open(output_path, "w", newline="") as file:
        write_rows = []
        # creating a csv writer object
        csvwriter = csv.writer(file)
        # writing the fields
        csvwriter.writerow(["sample"] + site_names)

        for filename in os.listdir(directory):
            new_dir = os.path.join(directory, filename)
            open_path = os.path.join(new_dir, f"{filename}.GC_corrected.coverage.tsv")
            data = pd.read_csv(open_path, delimiter="\t", quotechar='"')
            write_rows.append([filename] + data[col_name].to_list())

        # writing the data rows
        csvwriter.writerows(write_rows)
    
    # generate aggregate file (incl. clinical + ichorCNA data)
    a = pd.read_csv(agg_file_path)
    b = pd.read_csv(output_path)
    merged = a.merge(b, on='sample', how="left")
    output_path = output_path.replace('_coverage.csv', '')
    merged.to_csv(f"{output_path}_aggregate.csv", index=False)


def linear_fit_correction(site_names, input_file, output_file):
    agg_data = pd.read_csv(input_file)
    if "central" in input_file:
        tumor = pd.read_csv("tfx_correction/LuCaP_TFBS_Central-Mean.tsv", delimiter="\t")
        healthy = pd.read_csv("tfx_correction/HD_TFBS_Central-Mean.tsv", delimiter="\t")
    elif "mean" in input_file:
        tumor = pd.read_csv("tfx_correction/LuCaP_TFBS_Window-Mean.tsv", delimiter="\t")
        healthy = pd.read_csv("tfx_correction/HD_TFBS_Window-Mean.tsv", delimiter="\t")
    else:
        raise Exception("Input file not valid.")
    for site in site_names:
        poly1d_fn = tfx_correction(tumor, healthy, site)
        agg_data[site] = poly1d_fn(agg_data["TFx"])
    agg_data.to_csv(output_file, index=False)


def smooth_data(x, y, points=500):
    """Apply spline interpolation to smooth data"""
    spline = make_interp_spline(x, y, k=3)  # k=3 means cubic spline
    x_smooth = np.linspace(x.min(), x.max(), points)
    y_smooth = spline(x_smooth)
    return x_smooth, y_smooth


def plot_data_helper(group, plot_columns, label, color, linestyle='-', linewidth=2):
    Median = group.median().to_numpy()
    upper = np.percentile(group, 60, axis=0)
    lower = np.percentile(group, 40, axis=0)
    
    # Smooth the lines using spline interpolation
    plot_columns_smooth, Median_smooth = smooth_data(np.array(plot_columns), Median)
    _, upper_smooth = smooth_data(np.array(plot_columns), upper)
    _, lower_smooth = smooth_data(np.array(plot_columns), lower)
    
    plt.plot(plot_columns_smooth, Median_smooth, label=label, lw=linewidth, color=color, linestyle=linestyle)
    plt.fill_between(plot_columns_smooth, upper_smooth, lower_smooth, color=color, alpha=0.1)


# plot_columns = np.arange(-990, 991, 15)
plot_columns = np.arange(-75, 76, 15)
str_plot_columns = [str(m) for m in plot_columns]


def plot_data(data, site, group_by, ncomp_flag, thr=None, hd_data=None):
    # plt.figure(figsize=(16, 4))
    
    if ncomp_flag:
        group_A = data[(data['site_name'] == site) & (data[group_by].astype(np.float64) < thr)][str_plot_columns]
        group_B = data[(data['site_name'] == site) & (data[group_by].astype(np.float64) >= thr)][str_plot_columns]
    else:
        group_A = data[(data['site_name'] == site) & (data[group_by] == 'N')][str_plot_columns]
        group_B = data[(data['site_name'] == site) & (data[group_by] == 'Y')][str_plot_columns]

    plot_data_helper(group_A, plot_columns, label='Non-Responders', color='#1f77b4', linestyle='--', linewidth=2)
    plot_data_helper(group_B, plot_columns, label='Responders', color='#ff7f0e', linestyle='-', linewidth=2)
    
    # plt.xlabel("Distance from site", fontsize=14)
    # plt.ylabel("Normalized Coverage", fontsize=14)
    plt.title(site, fontsize=18)
    
    # Reduce the number of x-axis ticks
    # x_tick_interval = 500  # Set desired interval between x-axis labels
    # plt.xticks(np.arange(-1000, 1001, x_tick_interval))

    # Reduce the number of y-axis ticks (you can specify fewer y-ticks as needed)
    # y_tick_interval = 0.05  # Set the desired interval between y-axis labels
    # plt.yticks(np.arange(0.9, 1.01, y_tick_interval))  # Adjust based on the range of your data

    # plt.xlim(-975, 975) # Full window
    plt.xlim(-75, 75) # Small central window
    
    # Enable grid lines for better readability
    # plt.grid(True, which='both', linestyle='--', linewidth=0.5)

    # Save plot
    save_path = f"mannu_plots_6/{group_by}"
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    plt.savefig(f"{save_path}/{site}_central30.png", bbox_inches='tight')
    plt.close()


def run_mannwhitneyu(site_names, input_file, output_file, group_by, ncomp_flag, thr=None, group_by2=None):
    data = pd.read_csv(input_file)
    # tfx_data = data
    tfx_data = data[data["TFx"].astype(np.float64) >= TFx_threshold]
    # tfx_data = data[(data["TFx"].astype(np.float64) >= TFx_threshold) & (data["ploidy"].astype(np.float64) >= ploidy_threshold) & (data["mean_coverage"].astype(np.float64) >= cov_threshold)]
    # tfx_data = data[data["incl?"].astype(np.float64) == 1]

    TF_score = []

    for each_TF in site_names:
        if ncomp_flag:
            feature_snip = tfx_data[[group_by, each_TF]]
            a_group = feature_snip[feature_snip[group_by].astype(np.float64) < thr].dropna().values[:, 1]
            b_group = feature_snip[feature_snip[group_by].astype(np.float64) >= thr].dropna().values[:, 1]
        elif group_by2:
            feature_snip = tfx_data[[group_by, each_TF, group_by2]]
            a_group = feature_snip[feature_snip[group_by2] == 'N'].values[:, 1].astype(np.float64) # all cabazitaxel
            b_group = feature_snip[(feature_snip[group_by] == 'Y') & (feature_snip[group_by2] == 'Y')].values[:, 1].astype(np.float64) # docetaxel (responders only)
        else:
            feature_snip = tfx_data[[group_by, each_TF]]
            # TODO: 02/12/23 added a .dropna() or else it wouldn't run but not sure why I didn't have it before. It affected MannU results.
            a_group = feature_snip[feature_snip[group_by] == 'N'].values[:, 1].astype(np.float64)
            b_group = feature_snip[feature_snip[group_by] == 'Y'].values[:, 1].astype(np.float64)
        log2FC = np.log2(np.mean(b_group) / np.mean(a_group))
        mannU =  mannwhitneyu(a_group, b_group)
        try:
            anco = ancova(data=data, dv=each_TF, covar='TFx', between=group_by)
        except:
            anco = {"p-unc": [1]}
        pearson = stats.pearsonr(data["TFx"].dropna().values, data[each_TF].dropna().values)
        pearson_pvalue = np.around(pearson[1], 100)
        pearson_rho = np.around(pearson[0], 3)
        result = [each_TF, mannU[1], log2FC, anco["p-unc"][0], pearson_rho, pearson_pvalue]
        TF_score.append(result)


    TF_pval_df = pd.DataFrame(TF_score, columns = ["gene", "mannu_pvalue", "log2FC", "ancova_pvalue", "pearson_rho", "pearson_pvalue_(TFx correlation)"], index = None)

    # Multiple test (Bonferroni) correction -- more conservative than FDR (q-value)
    pvals = TF_pval_df['mannu_pvalue'].values
    multitest = sm.stats.multipletests(pvals, alpha=0.05, method='bonferroni', is_sorted=False, returnsorted=False)
    TF_pval_df['Bonferroni_Adjusted_pvalue'] = multitest[1]
    # TF_pval_df['Bonferonni_reject'] = multitest[0]

    # False Discovery Rate correction
    multi_test2 = fdrcorrection(pvals, alpha=0.10, method='indep', is_sorted=False)
    TF_pval_df['FDR_Adjusted_pvalue'] = multi_test2[1]
    # TF_pval_df['FDR_reject'] = multi_test2[0]

    TF_pval_df["(-)log10"] = np.log10(TF_pval_df["Bonferroni_Adjusted_pvalue"])
    TF_pval_df.to_csv(output_file + ".csv", index=False)

    # Set plot style using seaborn
    sns.set(style="whitegrid")

    # Set the figure size for better clarity
    plt.figure(figsize=(10, 7))

    # Scatter plot with custom aesthetics
    x = TF_pval_df["log2FC"]
    y = -np.log10(TF_pval_df["ancova_pvalue"])
    size = np.abs(x) * 50  # Adjust size of points based on log2FC values

    # Create a continuous color palette
    norm = plt.Normalize(vmin=min(y), vmax=max(y))
    palette = plt.cm.get_cmap("coolwarm")  # Or any other continuous colormap

    # Loop through each point and assign a color
    colors = []
    for xi, yi in zip(x, y):
        if yi > 1.5:  # Check if the point is significant
            if xi < 0:
                colors.append("1")  # Significant and log2FC < 0
            else:
                colors.append("2")  # Significant and log2FC > 0
        else:
            colors.append("3")  # Non-significant

    colors = np.array(colors)
    scatter = sns.scatterplot(x=x, y=y, hue=colors, size=np.abs(x) * 50,
                            sizes=(20, 200), palette=["tab:gray", "tab:orange", "tab:blue"],
                            edgecolor="k", legend=False)

    # plt.xlim(-0.10, 0.15)
    plt.xlim(-0.03, 0.02)
    plt.xlabel("")
    plt.ylabel("")
    # plt.xlabel(r"$\log_2$(Fold Change)", fontsize=14)
    # plt.ylabel(r"$-\log_{10}$(p-value)", fontsize=14)

    # Customize color bar
    # norm = plt.Normalize(0, max(y))
    # scalar = plt.cm.ScalarMappable(cmap=palette, norm=norm)
    # scalar.set_array([])
    # plt.colorbar(scalar)

    # Annotate significant genes
    # for i in range(TF_pval_df.shape[0]):
    #     if y[i] > 1.5:  # Threshold for annotation
    #         plt.text(x=x[i] - 0.0005,  # Adjust x to shift label left
    #                 y=y[i],
    #                 s=TF_pval_df['gene'].values[i].replace(".hg38", ""),
    #                 fontdict=dict(color='black', size=8),
    #                 ha='right')  # Keep alignment right for the text


    # Tight layout for better spacing
    plt.tight_layout()

    # num_colors = 2
    # cmap = plt.get_cmap('viridis', num_colors)
    # cmap.set_under('gray')
    # x = TF_pval_df["log2FC"]
    # y = -np.log2(TF_pval_df["ancova_pvalue"]) #  Setting this as mannu_pvalue for now. Better statistically to use FDR_Adjusted_pvalue
    # # plt.title(output_file)
    # plt.scatter(x, y, c=y, cmap=cmap, vmin=4, vmax=12)
    # plt.axhline(y = 4, color = 'b', linestyle = 'dashed')  
    # plt.colorbar(extend='min')
    # # plt.xlim(-0.1, 0.1)
    # plt.ylim(0, 12)
    # plt.xlabel("log2 Fold Change")
    # plt.ylabel("-log2 q-value")
    # for i in range(TF_pval_df.shape[0]):
    #     if y[i] > 8:
    #         plt.text(x=x[i],
    #                 y=y[i],
    #                 s=str(TF_pval_df['gene'].values[i]).replace(".hg38", ""),
    #                 fontdict=dict(color='black',size=4))
    plt.savefig(output_file + ".png", dpi=300)
    plt.close()

    thr = -0.001
    # visuz.GeneExpression.volcano(df=TF_pval_df, lfc='log2FC', pv='FDR_Adjusted_pvalue', sign_line=True, plotlegend=True, lfc_thr=[thr, thr], pv_thr=[0.05, 0.05], legendpos='upper right', legendanchor=(1.46,1), figname=output_file)


def createUnsupervisedHeatmap(select_sites, input_file, output_file):
    print("Creating heatmap based on site list: " + ", ".join(select_sites))
    
    # Load the data
    data = pd.read_csv(input_file)
    
    # Label responders
    data['responder'] = data.apply(label_responders, axis=1)
    
    # Select relevant columns
    df = data[["sample", "responder"] + select_sites]
    
    # Set 'sample' as the index
    df.set_index('sample', inplace=True)
    
    # Sort samples by responder status ('Y' comes first, then 'N')
    df.sort_values(by='responder', inplace=True, ascending=False)
    
    # Create a color palette for responders vs non-responders
    responder_colors = df['responder'].map({'Y': 'green', 'N': 'red'})
    
    # Drop the 'responder' column from the data
    df_clean = df.drop(columns=['responder'])
    
    # Apply quantile normalization
    df_normalized = pd.DataFrame(
        quantile_transform(df_clean, axis=0, copy=True, output_distribution='uniform'),
        index=df_clean.index,
        columns=df_clean.columns
    )
    
    # Set up the clustermap with the normalized data
    clustermap = sns.clustermap(
        df_normalized.transpose(), # Transposed data to have samples on the x-axis and TFs on the y-axis
        cmap='mako_r',  # You can adjust this palette if needed
        col_colors=responder_colors, # Color the columns based on responder status
        col_cluster=True,
        row_cluster=True,
        linewidths=0.5,  # Add gridlines for better readability
        annot=False  # Set True if you want to annotate the heatmap with values
    )

    # Save the plot
    plt.savefig(output_file + ".png", dpi=300, bbox_inches='tight')


def createSupervisedHeatmap(select_sites, input_file, output_file):
    print("Creating heatmap based on site list: " + ", ".join(select_sites))
    data = pd.read_csv(input_file)
    data['responder'] = data.apply(label_responders, axis=1)
    df = data[["sample", "responder", "TFx", "percent_change"] + select_sites]
    df.set_index('sample', inplace=True)
    df.sort_values(by=['responder', 'percent_change'], ascending=[False, True], inplace=True)

    responder_colors = df['responder'].map({'Y': 'green', 'N': 'red'})
    
    # Extract and normalize the tumor fraction and PSA percentage change
    tumor_fraction = df["TFx"]
    psa_percent_change = df["percent_change"]
    
    df_clean = df.drop(columns=['responder', 'TFx', 'percent_change'])

    # Apply quantile normalization
    df_normalized = pd.DataFrame(
        quantile_transform(df_clean, axis=0, output_distribution='uniform'),
        index=df_clean.index,
        columns=df_clean.columns
    )
    
    # Set up the clustermap
    g = sns.clustermap(
        df_normalized.transpose(),  # Transposed to have samples on x-axis
        cmap='vlag',
        col_cluster=False,
        row_cluster=True,
        yticklabels=True,
        cbar_pos=None,
        col_colors=responder_colors
    )
    g.ax_heatmap.set_xlabel("")  # Remove x-axis title
    g.ax_heatmap.set_xticklabels([])  # Remove x-axis labels
    
    cbar = g.figure.colorbar(g.ax_heatmap.collections[0], ax=g.ax_heatmap, orientation='vertical')
    cbar_position = cbar.ax.get_position()
    cbar.ax.set_position([cbar_position.x0 + 0.05, cbar_position.y0, cbar_position.width, cbar_position.height])

    # Fix: Get sample order from df_clean.index and convert to list
    sample_order = list(df_clean.index)  

    # Create new axes for tumor fraction and PSA change bars
    ax_heatmap = g.ax_heatmap
    ax_tumor_fraction = ax_heatmap.inset_axes([0, -0.1, 1, 0.05], transform=ax_heatmap.transAxes)
    ax_psa_change = ax_heatmap.inset_axes([0, -0.2, 1, 0.05], transform=ax_heatmap.transAxes)

    # Convert sample names to numeric positions for plotting
    x_positions = np.arange(len(sample_order))

    # Plot tumor fraction as a bar plot
    ax_tumor_fraction.bar(x_positions, tumor_fraction.loc[sample_order], color='black')
    ax_tumor_fraction.set_xlim(-0.5, len(sample_order) - 0.5)
    ax_tumor_fraction.set_ylim(0, 1)

    # Remove all labels
    ax_tumor_fraction.set_xticks([])
    ax_tumor_fraction.set_yticks([])
    ax_tumor_fraction.set_frame_on(False)

    # Plot PSA percentage change as a bar plot
    colors = ['#0072B2' if val > 0 else '#E69F00' for val in psa_percent_change.loc[sample_order]]
    ax_psa_change.bar(x_positions, psa_percent_change.loc[sample_order], color=colors)
    ax_psa_change.set_xlim(-0.5, len(sample_order) - 0.5)
    ax_psa_change.set_ylim(-100, 100)  # Adjust to full range of -100% to 100%

    # Remove all labels
    ax_psa_change.set_xticks([])
    ax_psa_change.set_yticks([])
    ax_psa_change.set_frame_on(False)

    # Adjust layout and save
    plt.savefig(output_file + ".png", dpi=300, bbox_inches='tight')


if __name__ == "__main__":
    site_names = pd.read_csv("sites.csv")["site_names"].dropna().to_list()

    agg_file_path = "/fh/fast/ha_g/user/dchen4/ichorCNA_analysis/ra223_aggregate_data.csv"
    # calculate_coverage('ra223_results', 'ra223_central_coverage.csv', 'central_coverage', site_names, agg_file_path)
    # calculate_coverage('ra223_results', 'ra223_mean_coverage.csv', 'mean_coverage', site_names, agg_file_path)
    # calculate_coverage('ra223_results', 'ra223_amplitude_coverage.csv', 'amplitude', site_names, agg_file_path)
    # linear_fit_correction(site_names, "ra223_central_aggregate.csv", "ra223_central_aggregate_(TFx corrected).csv")
    # linear_fit_correction(site_names, "ra223_mean_aggregate.csv", "ra223_mean_aggregate_(TFx corrected).csv")

    groups = [
        ("rad-cycles", True, 6),
        ("pre-LN", False),
        ("pre-visceral", False),
        ("<6-new-bone", False),
        ("<6-LNP", False),
        ("<6-VP", False),
        ("post1-new-bone", False),
        ("post1-LNP", False),
        ("post1-VP", False),
        ("post2-new-bone", False),
        ("post2-LNP", False),
        ("post2-VP", False),
    ]

    # for gro in groups:
    #     print(gro)
    #     run_mannwhitneyu(site_names, "ra223_central_aggregate.csv", f"mannu_results/ra223_centc_mannu_({gro[0]})", gro[0], gro[1], gro[2] if gro[1] else None)
    #     run_mannwhitneyu(site_names, "ra223_mean_aggregate.csv", f"mannu_results/ra223_meanc_mannu_({gro[0]})", gro[0], gro[1], gro[2] if gro[1] else None)
    #     run_mannwhitneyu(site_names, "ra223_amplitude_aggregate.csv", f"mannu_results/ra223_amp_mannu_({gro[0]})", gro[0], gro[1], gro[2] if gro[1] else None)

        # run_mannwhitneyu(site_names, "ra223_central_aggregate_(TFx corrected).csv", f"mannu_results_(TFx corrected)/ra223_centc_mannu_({gro[0]})", gro[0], gro[1], gro[2] if gro[1] else None)
        # run_mannwhitneyu(site_names, "ra223_mean_aggregate_(TFx corrected).csv", f"mannu_results_(TFx corrected)/ra223_meanc_mannu_({gro[0]})", gro[0], gro[1], gro[2] if gro[1] else None)


    # hd_data = pd.read_csv("sum_hd_results.csv", delimiter=",", quotechar='"')
    # cov_data = pd.read_csv("sum_ra223_results.csv", delimiter=",", quotechar='"')
    # for gro in groups:
    #     print(gro)
    #     clin_data = pd.read_csv("ra223_central_aggregate.csv", delimiter=",", quotechar='"')[['sample', gro[0]]]
    #     data = pd.merge(cov_data, clin_data, on='sample')
    #     for site in site_names:
    #         plot_data(data, site, gro[0], gro[1], gro[2] if gro[1] else None)




    agg_file_path = "/fh/fast/ha_g/user/dchen4/ichorCNA_analysis/doc_aggregate_data.csv"
    cbz_only = "/fh/fast/ha_g/user/dchen4/ichorCNA_analysis/cbz_only_aggregate_data.csv"
    dctx_only = "/fh/fast/ha_g/user/dchen4/ichorCNA_analysis/dctx_only_aggregate_data.csv"
    # responders and non-responders filtered samples
    dctx_RNR_only = "/fh/fast/ha_g/user/dchen4/ichorCNA_analysis/R_NR_IDs_PSA.csv"
    dctx_cbz_RNR = "/fh/fast/ha_g/user/dchen4/ichorCNA_analysis/R_NR_IDs_PSA_dctx+cbz.csv"

    # calculate_coverage('dctx_results_051523', 'cbz_only_central_coverage.csv', 'central_coverage', site_names, cbz_only)
    # calculate_coverage('dctx_results_051523', 'cbz_only_mean_coverage.csv', 'mean_coverage', site_names, cbz_only)
    # calculate_coverage('dctx_results_051523', 'dctx_only_central_coverage.csv', 'central_coverage', site_names, dctx_only)
    # calculate_coverage('dctx_results_051523', 'dctx_only_mean_coverage.csv', 'mean_coverage', site_names, dctx_only)

    # calculate_coverage('dctx_results_092124', '2dctx_RNR_only_central_coverage.csv', 'central_coverage', site_names, dctx_RNR_only)
    # calculate_coverage('dctx_results_092124', '2dctx_RNR_only_mean_coverage.csv', 'mean_coverage', site_names, dctx_RNR_only)
    # calculate_coverage('dctx_results_092124', '2dctx_cbz_RNR_central_coverage.csv', 'central_coverage', site_names, dctx_cbz_RNR)
    # calculate_coverage('dctx_results_092124', '2dctx_cbz_RNR_mean_coverage.csv', 'mean_coverage', site_names, dctx_cbz_RNR)

    # linear_fit_correction(site_names, "dctx_central_aggregate.csv", "dctx_central_aggregate_(TFx corrected).csv")
    # linear_fit_correction(site_names, "dctx_mean_aggregate.csv", "dctx_mean_aggregate_(TFx corrected).csv")

    groups = [
        # ("# prior tx", True, 1),
        # ("# cycles of docetaxel", True, 4),
        # ("prior secondary hormonal manipulation", False),
        # ("prior ketoconazole", False),
        # ("prior abiraterone progression", False),
        # ("prior enzalutamide progression", False),
    #     ("prior ra-223 progression", False), # NO DATA FOR 'Y' GROUP
        # ("bone involvement", False),
        # ("LN involvement", False),
        # ("visceral involvement", False),
        # ("days from sample withdrawal to start of docetaxel", True, 60),
        # ("PSA at docetaxel start", True, 4), # Serum PSA levels over 4 ng/mL is highly sensitive for prostate cancer
    #     ("% PSA decline", True, 0.91), # NO DATA FOR 'Y' GROUP (no ichorCNA data)
        # ("psa50", False),
        # ("PSA decline >= 80% within 16 weeks after docetaxel start", False),
        # ("radiologic response within 12 weeks from docetaxel start", False),
        # ("resistance", False),
        ("response", False),
    ]

    for gro in groups:
        print(gro)
        # run_mannwhitneyu(site_names, "2dctx_RNR_only_central_aggregate.csv", f"mannu_results_filtered/dctxRNR_centc_mannu_({gro[0]})_nolabels", gro[0], gro[1], None)
        # run_mannwhitneyu(site_names, "2dctx_RNR_only_mean_aggregate.csv", f"mannu_results_filtered/dctxRNR_meanc_mannu_({gro[0]})_nolabels", gro[0], gro[1], None)

        # run_mannwhitneyu(site_names, "2dctx_cbz_RNR_central_aggregate.csv", f"mannu_results_filtered/dctx_cbz_RNR_centc_mannu_({gro[0]})", gro[0], gro[1], gro[2] if gro[1] else None, group_by2='dctx')
        # run_mannwhitneyu(site_names, "2dctx_cbz_RNR_mean_aggregate.csv", f"mannu_results_filtered/dctx_cbz_RNR_meanc_mannu_({gro[0]})", gro[0], gro[1], gro[2] if gro[1] else None, group_by2='dctx')

    # cov_data = pd.read_csv("all_dctx_results_092124.csv", delimiter=",", quotechar='"')
    # for gro in groups:
    #     print(gro)
    #     clin_data = pd.read_csv("/fh/fast/ha_g/user/dchen4/ichorCNA_analysis/R_NR_IDs_PSA.csv", delimiter=",", quotechar='"')[['sample', gro[0]]]
    #     # clin_data[gro[0]] = clin_data[gro[0]].replace({'Y': 1, 'N': 0})
    #     # clin_data = clin_data[(clin_data["TFx"].astype(np.float64) >= TFx_threshold) & (clin_data["ploidy"].astype(np.float64) >= ploidy_threshold) & (clin_data["mean_coverage"].astype(np.float64) >= cov_threshold)][['sample', gro[0]]]
    #     # clin_data = clin_data[clin_data["incl?"].astype(np.float64) == 1]
    #     data = pd.merge(cov_data, clin_data, on='sample')
    #     print(data.head(10))
        
    #     for site in site_names:
    #         plot_data(data, site, gro[0], gro[1], gro[2] if gro[1] else None)


    # agg_file_path = "/fh/fast/ha_g/user/dchen4/ichorCNA_analysis/jci_clinical_data2.csv"
    # calculate_coverage('jci_results', 'jci_central_coverage.csv', 'central_coverage', site_names, agg_file_path)
    # calculate_coverage('jci_results', 'jci_mean_coverage.csv', 'mean_coverage', site_names, agg_file_path)

    select_TFs_to_show_central = pd.read_csv("custom_TF_list.csv")["site_names_central"].dropna().to_list()[:20]
    select_TFs_to_show_mean = pd.read_csv("custom_TF_list.csv")["site_names_mean"].dropna().to_list()[:20]

    createSupervisedHeatmap(select_TFs_to_show_central, "2dctx_RNR_only_central_aggregate.csv", "2dctx_RNR_only_sup-heatmap_central")
    createSupervisedHeatmap(select_TFs_to_show_mean, "2dctx_RNR_only_mean_aggregate.csv", "2dctx_RNR_only_sup-heatmap_mean")
    # createUnsupervisedHeatmap(select_TFs_to_show, "2dctx_RNR_only_central_aggregate.csv", "2dctx_RNR_only_unsup-heatmap_central")
    # createUnsupervisedHeatmap(select_TFs_to_show, "2dctx_RNR_only_mean_aggregate.csv", "2dctx_RNR_only_unsup-heatmap_mean")
    