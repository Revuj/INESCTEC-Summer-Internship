import pandas as pd
import numpy as np
from sklearn.decomposition import PCA


def main():
    dataframe = pd.read_csv("output2.csv")
    datamatrix = dataframe.values[:, 1:]
    pca = PCA(n_components=10)
    pca.fit(datamatrix)
    #PCA(copy=True, iterated_power='auto', n_components=10, random_state=None,
     #   svd_solver='auto', tol=0.0, whiten=False)


    dataframe = pd.DataFrame(data=pca.components_)
    dataframe.to_csv('output3.csv')
    np.savetxt("pca_components.csv", np.asarray(pca.components_), delimiter=",")
    np.savetxt("pca_components_abs.csv", abs(np.asarray(pca.components_)), delimiter=",")
    np.savetxt("pca_variance.csv", np.asarray(pca.explained_variance_ratio_), delimiter=",")


    metrics_weights = []
    for line in abs(pca.components_):
        total = sum(line)
        line_list = []
        for attribute in line:
            weight = attribute / total
            line_list.append(weight)
        metrics_weights.append(line_list)

    np.savetxt("metric_weights.csv", np.asarray(metrics_weights), delimiter=",")


if __name__ == '__main__':
    main()