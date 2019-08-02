import pandas as pd
import numpy as np


def main():
    """Reads the csv's outputed by normalize_ratios.py and calculates their variance.
    """
    with open('lib_metrics/scheme_principal.txt') as f:
        content = f.readlines()

    content = [x.strip() for x in content]

    files_names = os.listdir("normalized_ratios")

    for filename in files_names:
        diff = pd.read_csv("normalized_ratios/" + filename)
        same = pd.read_csv("normalized_ratios_same/" + filename)

        mean_same = []
        var_same = []
        mean_diff = []
        var_diff = []

        for metric in content:
            mean_same.append(same[metric].mean())
            var_same.append(same[metric].var())
            mean_diff.append(diff[metric].mean())
            var_diff.append(diff[metric].var())

        np.savetxt("comp/mean_same" + filename + ".csv", mean_same, delimiter=",", fmt='%s')
        np.savetxt("comp/var_same" + filename + ".csv", var_same, delimiter=",", fmt='%s')
        np.savetxt("comp/mean_diff" + filename + ".csv", mean_diff, delimiter=",", fmt='%s')
        np.savetxt("comp/var_diff" + filename + ".csv", var_diff, delimiter=",", fmt='%s')


if __name__ == '__main__':
    main()
