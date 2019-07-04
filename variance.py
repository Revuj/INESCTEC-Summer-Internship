import pandas as pd
import numpy as np
from sklearn.decomposition import PCA


def main():
    with open('lib_metrics/scheme_principal.txt') as f:
        content = f.readlines()

    content = [x.strip() for x in content]

    dataframe = pd.read_csv("output.csv")

    mean_list = []
    var_list = []

    for metric in content:
        mean_list.append(dataframe[metric].mean())
        var_list.append(dataframe[metric].var())

    np.savetxt("means.csv", mean_list, delimiter=",", fmt='%s')
    np.savetxt("variances.csv", var_list, delimiter=",", fmt='%s')


if __name__ == '__main__':
    main()