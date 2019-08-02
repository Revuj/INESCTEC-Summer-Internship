import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import os

def main():
    """Reads the csv's outputed by normalize_ratios.py and calculates their pca.
    """
    files_names = os.listdir("normalized_ratios_same")

    for filename in files_names:
        dataframe_same = pd.read_csv("normalized_ratios_same/" + filename)
        dataframe_diff = pd.read_csv("normalized_ratios_diff/" + filename)

        datamatrix_same = dataframe_same.values[:, 1:]
        pca_same = PCA(n_components=10)
        pca_same.fit(datamatrix_same)

        datamatrix_diff = dataframe_diff.values[:, 1:]
        pca_diff = PCA(n_components=10)
        pca_diff.fit(datamatrix_diff)


        np.savetxt("pca_same/pca_components" + filename, np.asarray(pca.components_), delimiter=",")
        np.savetxt("pca_same/pca_variance" + filename, np.asarray(pca.explained_variance_ratio_), delimiter=",")
        np.savetxt("pca_diff/pca_components" + filename, np.asarray(pca.components_), delimiter=",")
        np.savetxt("pca_diff/pca_variance" + filename, np.asarray(pca.explained_variance_ratio_), delimiter=",")




if __name__ == '__main__':
    main()
