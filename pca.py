import pandas as pd
from sklearn.decomposition import PCA


def main():
    dataframe = pd.read_csv("output2.csv")
    datamatrix = dataframe.values[:, 1:]
    pca = PCA(n_components=10)
    pca.fit(datamatrix)

    components = pd.DataFrame(data=pca.components_)
    components.to_csv('pca_components.csv')


    variance_ration = pd.DataFrame(data=pca.explained_variance_ratio_)
    variance_ration.to_csv('explained_variance_racio.csv')



if __name__ == '__main__':
    main()