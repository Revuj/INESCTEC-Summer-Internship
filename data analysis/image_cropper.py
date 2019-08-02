import pandas as pd
import cv2
from xml.dom import minidom
import os
import dlib
import xml.etree.ElementTree as ET


def main():
    """Runs through a folder with images, reads their respective xml (generated with face_detector.py or xml_generator.py) and crops the image in a way that only the face stays.
    """
    detector = dlib.get_frontal_face_detector()
    tl = []
    br = []
    count = 0
    folders = os.listdir("dataset")

    for foldername in folders:
        filenames = os.listdir("dataset/" + foldername)
        print("beginning folder " + foldername)
        for image in filenames:
            print(image[-4:] )
            if (image[-4:] == '.jpg'):
                numpyimg = cv2.imread("dataset/" + foldername + "/" + image)
                print("dataset/" + foldername + "/" + image)
                print(numpyimg)
                tree = ET.parse("dataset/" + foldername + "/" + image[:-4] + '.xml')
                root = tree.getroot()
                x1 = root[6][4][0].text
                y1 = root[6][4][1].text
                x2 = root[6][4][2].text
                y2 = root[6][4][3].text
                tl.append((int(x1), int(y1)))
                br.append((int(x2), int(y2)))

                print(x1 + " " + y1 + " " + x2 + " " + y2)
                numpyimg = numpyimg[int(y1):int(y2), int(x1):int(x2)]
                cv2.imwrite("dataset/" + foldername + "/" + image, numpyimg)
                tl = []
                br = []




if __name__ == '__main__':
    main()
