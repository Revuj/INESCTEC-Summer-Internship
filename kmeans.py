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



    files_names = os.listdir("normalized_ratios")
    index = 0
    for filename in files_names:
        image_counter = 0
        dict = {}
        while image_counter < 50:
            dataframe = pd.read_csv("normalized_ratios/" + filename)
            datamatrix = dataframe.values[:, 1:]

            print(filename)
            labels = pd.read_csv("landmarks/out" + filename[10:])

            kmeans = KMeans(init='k-means++', n_clusters=50).fit(datamatrix)

            new_dataframe = pd.DataFrame({"filepath": [], "label": []})
            print('#########' + filename + '#########')
            for centroid in kmeans.cluster_centers_:
                min_distance = 100.0
                counter = 0
                min_index = 0
                for image in datamatrix:
                    dist = distance.euclidean(image, centroid)
                    if dist < float(min_distance):
                        min_distance = dist
                        min_index = counter
                    counter += 1

                if filename[10:-4] != '1' and filename[10:-4] != '9213':
                    img = cv2.imread("dataset/" + map[filename[10:-4]] + "/diff/" + labels['filename'][min_index][15:],
                                     -1)

                    if labels['filename'][min_index][15:] in dict:
                        print("already exists")
                        dict[labels['filename'][min_index][15:]] += 1
                    else:
                        dict.update({labels['filename'][min_index][15:]: 1})

                    if (dict[labels['filename'][min_index][15:]] == 5):
                        image_counter += 1
                        cv2.imwrite("kmeans50/" + filename[10:-4] + "/" + labels['filename'][min_index][15:], img)
                        new_dataframe = new_dataframe.append(
                        {'filepath': labels['filename'][min_index][15:], 'label': map[filename[10:-4]]},
                        ignore_index=True)
                else:
                    img = cv2.imread("dataset/" + map[filename[10:-4]] + "/diff/" + labels['filename'][min_index][8:],
                                     -1)

                    if labels['filename'][min_index][8:] in dict:
                        dict[labels['filename'][min_index][8:]] += 1
                    else:
                        dict.update({labels['filename'][min_index][8:]: 1})

                    if (dict[labels['filename'][min_index][8:]] == 5):
                        image_counter += 1
                        cv2.imwrite("kmeans50/" + filename[10:-4] + "/" + labels['filename'][min_index][8:], img)
                        new_dataframe = new_dataframe.append(
                        {'filepath': labels['filename'][min_index][8:], 'label': map[filename[10:-4]]},
                        ignore_index=True)

                index += 1
                new_dataframe.to_csv(map[filename[10:-4]] + "kmeans50.csv")


if __name__ == '__main__':
    main()
