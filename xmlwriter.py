import pandas as pd
import cv2
import xml.etree.ElementTree as ET
from xml.dom import minidom
import os
import dlib
import sys


def write_csv():
    people = os.listdir("dataset")
    for person in people:
        datasetTypes = os.listdir("dataset/" + person)
        for datasetType in datasetTypes:
            if (datasetType[-3:] != 'csv'):
                images = pd.DataFrame(columns = ["filepath", "label"])
                imageList = os.listdir("dataset/" + person + "/" + datasetType)
                for image in imageList:
                    if (image[-3:] !='xml'):
                        #print("copying " + image)
                        images = images.append({"filepath" : image, "label": str(person)}, ignore_index = True)


                images.to_csv("dataset" + "/" + person + "/" + datasetType +"3.csv" , sep=",")

def main():
    write_csv()

if __name__ == '__main__':
    main()