# @Author: Luis Vilaca <lvilaca>
# @Date:   2019-04-23T12:04:57+01:00
# @Email:  luis.m.salgado@inesctec.pt
# @Filename: landmark_detection.py
# @Last modified by:   lvilaca
# @Last modified time: 2019-04-23T13:26:35+01:00
# Example call: python3 landmark_detection.py -path_model ./shape_predictor_68_face_landmarks.dat -path data/ruimoreira_diff

import sys
import os
import dlib
import glob
import cv2
import pandas as pd
import xml.etree.ElementTree as ET
import math
import numpy as np
import argparse


def union(au, bu, area_intersection):
    area_a = (au[2] - au[0]) * (au[3] - au[1])
    area_b = (bu[2] - bu[0]) * (bu[3] - bu[1])
    area_union = area_a + area_b - area_intersection

    return area_union


def intersection(ai, bi):
    x = max(ai[0], bi[0])
    y = max(ai[1], bi[1])
    w = min(ai[2], bi[2]) - x
    h = min(ai[3], bi[3]) - y

    if w < 0 or h < 0:
        return 0

    return w*h


def iou(a, b):
    if a[0] >= a[2] or a[1] >= a[3] or b[0] >= b[2] or b[1] >= b[3]:
        return 0.0

    area_i = intersection(a, b)
    area_u = union(a, b, area_i)

    return float(area_i)/float(area_u + 1e-6)


def calc_dist(p1, p2):
    return math.sqrt(math.pow((p1[0]-p2[0]), 2)+math.pow((p1[1]-p2[1]), 2))


def parse_gt(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    for member in root.findall('object'):
        value = [int(member.find('bndbox')[0].text),
                 int(member.find('bndbox')[1].text),
                 int(member.find('bndbox')[2].text),
                 int(member.find('bndbox')[3].text)]
        return value


def calc_metrics(point_list, bbox):
    scheme_list = 'lib_metrics/scheme1.txt'

    result_dict = {}
    # Load metrics and idx from txt
    with open(scheme_list, 'r') as inf:
        dict_from_file = eval(inf.read())

    for key, value in dict_from_file.items():
        if(key.find('tn') < 2 and key.find('tn') != -1):
            p1 = tuple((point_list[int(value[0])-1][0], int(bbox[1])))
            p2 = tuple(point_list[int(value[1])-1])
            dist = calc_dist(p1, p2)

        elif(key.find('tn') > 2 and key.find('tn') != -1):
            p1 = tuple((point_list[int(value[1])-1][0], int(bbox[1])))
            p2 = tuple(point_list[int(value[0])-1])
            dist = calc_dist(p1, p2)

        else:
            p1 = tuple(point_list[int(value[0])-1])
            p2 = tuple(point_list[int(value[1])-1])
            dist = calc_dist(p1, p2)

        result_dict[key] = round(dist, 3)

    return result_dict


def calc_ratios(measures_dict):
    ratios_label = [line.rstrip('\n') for line in open('lib_metrics/ratios.txt')]
    ratios_dict = {}
    for label in ratios_label:
        div = [x.strip() for x in label.split(',')]

        if div[0] == 'ps-pi':
            num = (measures_dict['ps-pi1']+measures_dict['ps-pi2'])/2
            ratio = float(num)/float(measures_dict[div[1]])
        elif div[1] == 'ps-pi':
            denom = (measures_dict['ps-pi1']+measures_dict['ps-pi2'])/2
            ratio = float(measures_dict[div[0]])/float(denom)
        else:
            ratio = float(measures_dict[div[0]])/float(measures_dict[div[1]])

        key = '{}-{}'.format(div[0], div[1])
        ratios_dict[key] = round(ratio, 3)

    return ratios_dict


def get_mag_ang(img):
    """Gets image gradient (magnitude) and orientation (angle).

    Parameters
    ----------
    img

    Returns
    -------
    Gradient & Orientation

    """

    img = np.sqrt(img)

    gx = cv2.Sobel(np.float32(img), cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(np.float32(img), cv2.CV_32F, 0, 1)

    mag, ang = cv2.cartToPolar(gx, gy)

    return mag, ang


def calc_DD_and_EOS(mag_l, ang_l, mag_r, ang_r):
    mag = mag_l - mag_r
    mag = mag.reshape(1, -1)
    ang = np.cos(mag)
    # Mean
    mag = np.mean(mag)
    ang = np.mean(ang)

    return round(mag, 3), round(ang, 3)


def calc_symmetry(img, bbox, point_list):
    symmetry_dict = {}
    #C1 = tuple((point_list[39][0]-bbox[0], point_list[39][1]-bbox[1]))
    #C2 = tuple((point_list[42][0]-bbox[0], point_list[42][1]-bbox[1]))
    C3 = tuple((point_list[33][0]-bbox[0], point_list[33][1]-bbox[1]))

    assert img is not None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    cropped = img[bbox[1]:bbox[3], bbox[0]:bbox[2]]
    # Divide the image
    cropped_left = cropped[:, :C3[0]]
    cropped_right = cropped[:, C3[0]:]
    # compare widths to resize
    h, wl = cropped_left.shape
    h, wr = cropped_right.shape

    if wl > wr:
        cropped_right = cv2.resize(cropped_right, (wl, h), interpolation=cv2.INTER_AREA)
    elif wl < wr:
        cropped_left = cv2.resize(cropped_left, (wr, h), interpolation=cv2.INTER_AREA)

    # Calc magnitude and orientation
    mag_left, ang_left = get_mag_ang(cropped_left)
    mag_right, ang_right = get_mag_ang(cropped_right)
    # Calc all
    mag, ang = calc_DD_and_EOS(mag_left, ang_left, mag_right, ang_right)
    symmetry_dict['DD'] = mag
    symmetry_dict['EOS'] = ang

    return symmetry_dict


def calc_contrast(inner_list, outer_list):
    return round((np.sum(outer_list)-np.sum(inner_list))/(np.sum(outer_list)+np.sum(inner_list)), 3)


def calc_region_contrast(img, points, file):
    results_contrast = {}
    label_regions = ['Cl_lips', 'Ca_lips', 'Cb_lips', 'Cl_eyes', 'Ca_eyes', 'Cb_eyes',
        'Cl_eyebrows', 'Ca_eyebrows', 'Cb_eyebrows']
    # Color conversion
    img_Lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    # eyebrows, lips, eyes
    x_points = [point[0] for point in points]
    y_points = [point[1] for point in points]

    region_dict = {
        'eyes': [(37, 42), (43, 48)],
        'lips': [(49, 60)],
        'eyebrows': [(18, 22), (23, 27)],
    }

    for e, key in enumerate(region_dict.keys()):
        for count, region in enumerate(region_dict[key]):

            xmax = max(x_points[region[0]-1:region[1]-1])
            xmin = min(x_points[region[0]-1:region[1]-1])
            ymax = max(y_points[region[0]-1:region[1]-1])
            ymin = min(y_points[region[0]-1:region[1]-1])

            # Lab image crop
            cropped_img_inner = img_Lab[ymin:ymax, xmin:xmax]
            cropped_img_outer = img_Lab[int(ymin*0.975):int(ymax*1.025),
                int(xmin*0.975):int(xmax*1.025)]

            for i in range(0, 3, 1):
                contrast = calc_contrast(cropped_img_inner[:, :, i].reshape(1, -1),
                    cropped_img_outer[:, :, i].reshape(1, -1))
                assert contrast <= 1, '{}'.format(file)

                # Add to dict with labels from label_regions
                results_contrast[label_regions[(e*3)+i]] = contrast
                if(count > 0):
                    results_contrast[label_regions[(e*3)+i]] += contrast
                    results_contrast[label_regions[(e*3)+i]] /= (count+1)

    return results_contrast


def main():
    parser = argparse.ArgumentParser(description='Diversity in faces dataset generator.')
    parser.add_argument('-path', '--path_to_folder', required=True)
    parser.add_argument('-path_model', '--path_to_model', default='shape_predictor_68_face_landmarks.dat')
    parser.add_argument('-out', '--out_path', required=True)
    args = vars(parser.parse_args())

    predictor_path = args['path_to_model']
    faces_folder_path = args['path_to_folder']
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)

    # Init
    idx_label = [line.rstrip('\n') for line in open('lib_metrics/scheme_all.txt')]
    dataframe_points = pd.DataFrame(columns=idx_label)
    final_str = []
    points = []
    idx = 0

    for f in glob.glob(os.path.join(faces_folder_path, "*.jpg")):
        # Ground truth
        sys.stdout.write('count: {}\r'.format(idx))
        sys.stdout.flush()
        gt_box = parse_gt(f.replace('.jpg', '.xml'))
        # Load img and run facial detector
        img = cv2.imread(f)
        dets = detector(img, 1)

        for k, d in enumerate(dets):
            bbox = [int(d.left()), int(d.top()), int(d.right()), int(d.bottom())]
            for idx_b, b in enumerate(bbox):
                if b < 0:
                    bbox[idx_b] = 0

            # face is relevant?
            if(iou(bbox, gt_box) > 0.1):
                final_str[:] = []
                points[:] = []

                # find landmarks
                shape = predictor(img, d)

                for x in range(68):
                    point = (int(shape.part(x).x), int(shape.part(x).y))
                    points.append(point)

                # Calc each metric
                metrics_dict = calc_metrics(points, bbox)
                ratios_dict = calc_ratios(metrics_dict)
                symmetry_dict = calc_symmetry(img, bbox, points)
                contrast_dict = calc_region_contrast(img, points, f)
                # Append results to dataframe
                final_str.append(f)
                for val in metrics_dict.values(): final_str.append(val)
                for rat in ratios_dict.values(): final_str.append(rat)
                for sym in symmetry_dict.values(): final_str.append(sym)
                for contrast in contrast_dict.values(): final_str.append(contrast)

                dataframe_points.loc[idx] = final_str
                idx += 1

    dataframe_points.to_csv(args['out_path'])


if __name__ == '__main__':
    main()


