import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import os

def main():
    files_names = os.listdir("normalized_ratios")

    for filename in files_names:
        dataframe = pd.read_csv("normalized_ratios/" + filename)
        datamatrix = dataframe.values[:, 1:]
        pca = PCA(n_components=10)
        pca.fit(datamatrix)
        #PCA(copy=True, iterated_power='auto', n_components=10, random_state=None,
         #   svd_solver='auto', tol=0.0, whiten=False)


        np.savetxt("pca/pca_components" + filename[10:], np.asarray(pca.components_), delimiter=",")
        np.savetxt("pca/pca_variance" + filename[10:], np.asarray(pca.explained_variance_ratio_), delimiter=",")




if __name__ == '__main__':
    main()