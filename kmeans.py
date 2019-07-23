import os

import cv2
import pandas as pd
from scipy.spatial import distance
from sklearn.cluster import KMeans


def main():
    map = {'1': 'Dalai_Lama',
           '29': 'Abdullah_II_of_Jordan',
           '82': 'Aditya_Seal',
           '149': 'Aishwarya_Rai',
           '178': 'Alain_Traore',
           '259': 'Alex_Gonzaga',
           '596': 'Angelique_Kidjo',
           '658': 'Anne_Princess_Royal',
           '736': 'Cavaco_Silva',
           '838': 'Aya_Miyama',
           '1781': 'Conan_O_Brian',
           '9213': 'Zelia_Duncan',
           }

    train_dataframe = pd.DataFrame({"filepath": [], "label": []})
    val_dataframe = pd.DataFrame({"filepath": [], "label": []})
    files_names = os.listdir("normalized_ratios")
    index = 0
    for filename in files_names:
        image_counter = 0
        dict = {}
        dataframe = pd.read_csv("normalized_ratios/" + filename)
        datamatrix = dataframe.values[:, 1:]

        labels = pd.read_csv("landmarks/out" + filename[10:])

        for index, row in labels.iterrows():
            if filename[10:-4] != '1' and filename[10:-4] != '9213':
                dict.update({row['filename'][15:]: 0})
            else:
                dict.update({row['filename'][8:]: 0})

        print(len(dict.items()))

        while image_counter < 100:
            kmeans = KMeans(init='k-means++', n_clusters=100).fit(datamatrix)

            print('#########' + filename + '#########')
            for centroid in kmeans.cluster_centers_:
                min_distance = 20000.0
                counter = 0
                min_index = 0
                for image in datamatrix:
                    dist = distance.euclidean(image, centroid)
                    if dist < float(min_distance):
                        min_distance = dist
                        min_index = counter
                    counter += 1

                if filename[10:-4] != '1' and filename[10:-4] != '9213':
                    #img = cv2.imread("dataset/" + map[filename[10:-4]] + "/diff/" + labels['filename'][min_index][15:], -1)

                    dict[labels['filename'][min_index][15:]] += 1
                    if (dict[labels['filename'][min_index][15:]] == 5):
                        image_counter += 1
                        #cv2.imwrite("kmeans100/" + filename[10:-4] + "/" + labels['filename'][min_index][15:], img)

                else:
                    #img = cv2.imread("dataset/" + map[filename[10:-4]] + "/diff/" + labels['filename'][min_index][8:], -1)

                    dict[labels['filename'][min_index][8:]] += 1

                    if (dict[labels['filename'][min_index][8:]] == 5):
                        image_counter += 1
                        #cv2.imwrite("kmeans100/" + filename[10:-4] + "/" + labels['filename'][min_index][8:], img)

                index += 1

        print(len(dict))
        for img, cnt in dict.items():
            if cnt >= 5:
                train_dataframe = train_dataframe.append({'filepath': map[filename[10:-4]] + "_" + img, 'label': map[filename[10:-4]]}, ignore_index=True)
            else:
                val_dataframe = val_dataframe.append({'filepath': map[filename[10:-4]] + "_" + img, 'label': map[filename[10:-4]]}, ignore_index=True)

    train_dataframe.to_csv("kmeans100/train.csv")
    val_dataframe.to_csv("kmeans100/val.csv")



if __name__ == '__main__':
    main()
