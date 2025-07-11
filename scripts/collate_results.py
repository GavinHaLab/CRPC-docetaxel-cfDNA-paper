from ast import Index
import csv
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def collate_results(directory, file_name, headers):
    with open(file_name, "w", newline="") as file:
        # creating a csv writer object
        csvwriter = csv.writer(file)
        # writing the fields
        csvwriter.writerow(headers)

        for filename in os.listdir(directory):
            new_dir = os.path.join(directory, filename)
            open_path = os.path.join(new_dir, f"{filename}.GC_corrected.coverage.tsv")
            data = pd.read_csv(open_path, delimiter="\t", quotechar='"', header=None, skiprows=1)
            # print(data.values.tolist())
            csvwriter.writerows(data.values.tolist())


def tfx_correction(tumor, healthy, site):
    site = site.replace(".hg38", "")
    t_group = tumor[site].to_list()
    h_group = healthy[site].to_list()
    y = t_group + h_group
    x = ([1] * len(t_group)) + ([0] * len(h_group))
    coef = np.polyfit(x,y,1)
    poly1d_fn = np.poly1d(coef) # poly1d_fn is now a function which takes in x and returns an estimate for y
    return poly1d_fn

    plt.plot(x,y, 'yo', x, poly1d_fn(x), '--k') #'--k'=black dashed line, 'yo' = yellow circle marker

    # plt.xlim(0, 5)
    # plt.ylim(0, 12)
    plt.xlabel("Tumor Fraction")
    plt.ylabel("Griffin Metric")
    plt.savefig('test.png')
    plt.close()


def collate_hd_data(hd_path, headers2):
    frames = []
    for filename in os.listdir(hd_path):
        new_dir = os.path.join(hd_path, filename)
        data = pd.read_csv(new_dir, delimiter="\t", quotechar='"')
        frames.append(data)
    result = pd.concat(frames)
    result = result[headers2]
    print(result)
    result.to_csv("sum_hd_results.csv", index=False)


if __name__ == "__main__":
    directory = 'dctx_results_092124'
    headers = pd.read_csv("headers.csv", quotechar='"')["headers"].to_list()
    collate_results(directory, "all_dctx_results_092124.csv", headers)
    # hd_path = "/fh/fast/ha_g/user/rpatton/HD_data/Griffin_TFBS_140-250bp/results/coverage/all_sites"
    # headers2 = pd.read_csv("headers.csv", quotechar='"')["headers2"].dropna().to_list()
    # collate_hd_data(hd_path, headers2)
