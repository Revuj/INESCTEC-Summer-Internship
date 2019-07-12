import pandas as pd
import cv2
import xml.etree.ElementTree as ET
from xml.dom import minidom
import os
import dlib
import sys
import scipy.misc
from PIL import Image

def write_xml(img_path, obj, tl, br):
    image = cv2.imread(img_path)
    head_img, tail_img = os.path.split(img_path)
    height, width, depth = image.shape

    if img_path[-3:] == 'jpg':
        savedir = os.path.join(head_img, tail_img.replace('.jpg', '.xml'))
    else:
        savedir = os.path.join(head_img, tail_img.replace('.png', '.xml'))

    annotation = ET.Element('annotation')
    ET.SubElement(annotation, 'folder').text = head_img
    ET.SubElement(annotation, 'filename').text = tail_img
    ET.SubElement(annotation, 'path').text = img_path

    source = ET.SubElement(annotation, 'source')
    ET.SubElement(source, 'database').text = 'Unknown'

    size = ET.SubElement(annotation, 'size')
    ET.SubElement(size, 'width').text = str(width)
    ET.SubElement(size, 'height').text = str(height)

    ET.SubElement(annotation, 'segmented').text = '0'

    for obj, topl, botr in zip(obj, tl, br):
        ob = ET.SubElement(annotation, 'object')
        ET.SubElement(ob, 'name').text = obj.title()
        ET.SubElement(ob, 'pose').text = 'Unspecified'
        ET.SubElement(ob, 'truncated').text = '0'
        ET.SubElement(ob, 'difficult').text = '0'
        bbox = ET.SubElement(ob, 'bndbox')
        ET.SubElement(bbox, 'xmin').text = str(topl[0])
        ET.SubElement(bbox, 'ymin').text = str(topl[1])
        ET.SubElement(bbox, 'xmax').text = str(botr[0])
        ET.SubElement(bbox, 'ymax').text = str(botr[1])

    xml_str = ET.tostring(annotation)
    root = ET.fromstring(xml_str)
    xml_str = minidom.parseString(ET.tostring(root)).toprettyxml(indent = "   ")

    xml_str_b = bytes(xml_str, 'utf-8')
    with open(savedir, 'wb') as temp_xml:
        temp_xml.write(xml_str_b)


def main():
    detector = dlib.get_frontal_face_detector()
    tl = []
    br = []
    count = 0
    folders = os.listdir("dataset")

    for foldername in folders:
        filenames = os.listdir("dataset/" +foldername)
        print("beginning folder " + foldername)
        for data in filenames:
            images = os.listdir("dataset/" + foldername + "/" + data)
            if data == 'test':
                for filename in images:
                    if filename[-3:] == "png" or filename[-3:] == "jpg":
                        numpyimg = cv2.imread("dataset/" + foldername + "/" + data + "/" + filename)
                        img = dlib.load_rgb_image("dataset/" + foldername + "/" + data + "/" + filename)
                        dets = detector(img, 1)

                        for (i, d) in enumerate(dets):
                            x1 = d.left()
                            y1 = d.top()
                            x2 = d.right()
                            y2 = d.bottom()

                        tl.append((x1, y1-20))
                        br.append((x2, y2+20))
                        write_xml("dataset/" + foldername + "/" + data + "/" + filename, [filename], tl, br)
                        numpyimg = numpyimg[y1:y2, x1:x2]
                        im = Image.fromarray(numpyimg)
                        im.save("dataset/" + foldername + "/" + data + "/" + filename)
                        tl = []
                        br = []
                        count += 1
            if data == 'diff':
                dataframe = pd.read_csv("loose_bb_test.csv")
                for filename in images:
                    tl = []
                    br = []
                    for idx, row in dataframe.iterrows():
                        if row['NAME_ID'].find(foldername + "/" + filename[:-4]) != -1:
                            numpyimg = cv2.imread("dataset/" + foldername + "/" + data + "/" + filename)
                            tlx = int(row['X'])
                            tly = int(row['Y'])
                            brx = int(row['X'] + row['W'])
                            bry = int(row['Y'] + row['H'])
                            tl.append((tlx, tly))
                            br.append((brx, bry))
                            write_xml("dataset/" + foldername + "/" + data + "/" + filename, [filename], tl, br)
                            numpyimg = numpyimg[tly:bry, tlx:brx]
                            im = Image.fromarray(numpyimg)
                            im.save("dataset/" + foldername + "/" + data + "/" + filename)
                            tl[:] = []
                            br[:] = []
                            break


if __name__ == '__main__':
    main()