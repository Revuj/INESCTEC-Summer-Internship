import pandas as pd
import cv2
import xml.etree.ElementTree as ET
from xml.dom import minidom
import os



def write_xml(img_path, obj, tl, br):
    image = cv2.imread(img_path)
    head_img, tail_img = os.path.split(img_path)
    height, width, depth = image.shape

    savedir = os.path.join(head_img, tail_img.replace('.jpg', '.xml'))

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
    dataframe = pd.read_csv("loose_bb_test.csv")
    print(os.listdir("n000001"))
    print(dataframe['NAME_ID'])

    obj = ["n000001"]
    tl =[]
    br = []
    for idx, row in dataframe.iterrows():
        if row['NAME_ID'].find("n000001") != -1 and os.path.isfile(row['NAME_ID'] + ".jpg"):
            tlx = int(row['X'])
            tly = int(row['Y'])
            brx = int(row['X'] + row['W'])
            bry = int(row['Y'] + row['H'])
            tl.append((tlx, tly))
            br.append((brx, bry))
            write_xml(row['NAME_ID'] + ".jpg", obj, tl, br)
            tl[:] = []
            br[:] = []



if __name__ == '__main__':
    main()