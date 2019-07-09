import pandas as pd
import numpy as np


def main():
    with open('lib_metrics/scheme_principal.txt') as f:
        content = f.readlines()

    content = [x.strip() for x in content]

    same = pd.read_csv("normalized_ratios/normalized178.csv")
    diff = pd.read_csv("normalized_ratios_same/178.csv")

    mean_same = []
    var_same = []
    mean_diff = []
    var_diff = []

    for metric in content:
        mean_same.append(same[metric].mean())
        var_same.append(same[metric].var())
        mean_diff.append(diff[metric].mean())
        var_diff.append(diff[metric].var())

    np.savetxt("comp/mean_sam178.csv", mean_same, delimiter=",", fmt='%s')
    np.savetxt("comp/var_sam178.csv", var_same, delimiter=",", fmt='%s')
    np.savetxt("comp/mean_dif178.csv", mean_diff, delimiter=",", fmt='%s')
    np.savetxt("comp/var_dif178.csv", var_diff, delimiter=",", fmt='%s')


if __name__ == '__main__':
    main()