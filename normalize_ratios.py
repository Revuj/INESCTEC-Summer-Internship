# @Author: Luís Vilaça <lvilaca>
# @Date:   2019-06-27T11:08:01+01:00
# @Email:  luis.m.salgado@inesctec.pt
# @Filename: normalize_ratios.py
# @Last modified by:   lvilaca
# @Last modified time: 2019-06-27T11:08:01+01:00


import pandas as pd
import numpy as np
import argparse
import math


def normalize_bbox(df, idx_label, args):
    normalized_df = pd.DataFrame(columns=idx_label)
    bboxnorm_label = [line.rstrip('\n') for line in open('lib_metrics/scheme_bboxnorm.txt')]
    ratios_list = [line.rstrip('\n') for line in open('lib_metrics/ratios.txt')]

    for idx, row in df.iterrows():
        if('zy-zy' in idx_label and 'tn-gn' in idx_label):
            hipot = math.sqrt(math.pow(float(row['zy-zy']), 2)+math.pow(float(row['tn-gn']), 2))
        row_norm = []
        for label in idx_label:
            if(label in bboxnorm_label):
                row_norm.append(round((abs(row[label])/hipot), 3))
            elif(label in ratios_list):
                div = [x.strip() for x in label.split(',')]

                if div[0] == 'ps-pi':
                    num = (round((abs(row['ps-pi1'])/hipot), 3)+round((abs(row['ps-pi2'])/hipot), 3))/2
                    ratio = float(num)/float(round((abs(row[div[1]])/hipot), 3))

                elif div[1] == 'ps-pi':
                    denom = (round((abs(row['ps-pi1'])/hipot), 3)+round((abs(row['ps-pi2'])/hipot), 3))/2
                    ratio = float(round((abs(row[div[1]])/hipot), 3))/float(denom)

                else:
                    hipot = math.sqrt(math.pow(float(row['zy-zy']), 2)+math.pow(float(row['tn-gn']), 2))
                    ratio = float(round((abs(row[div[0]])/hipot), 3))/float(round((abs(row[div[1]])/hipot), 3))

                row_norm.append(round(ratio, 3))
            else:
                row_norm.append(round(row[label], 3))

        normalized_df.loc[idx] = row_norm

    if('zy-zy' in idx_label and 'tn-gn' in idx_label):
        for ratios in ratios_list:
            max = normalized_df[ratios][np.argmax(normalized_df[ratios])]
            for idx, row in enumerate(normalized_df[ratios]):
                if(max > 1):
                    normalized_df[ratios].iloc[int(idx)] = round(normalized_df[ratios].iloc[int(idx)]/max, 3)

    #print(normalized_df.head())
    normalized_df.to_csv(args['out_final'])


def main():
    parser = argparse.ArgumentParser(description='Normalize ratios using the diagonal of the bbox.')
    parser.add_argument('-path', '--path_to_csv', required=True)
    parser.add_argument('-scheme', '--path_to_scheme_all', required=True)
    parser.add_argument('-scheme_final', '--path_to_final', required=True)
    parser.add_argument('-out_final', '--out_final', required=True)
    args = vars(parser.parse_args())

    idx_label = [line.rstrip('\n') for line in open(args['path_to_final'])]
    df = pd.read_csv(args['path_to_csv'])
    normalize_bbox(df, idx_label, args)


if __name__ == '__main__':
    main()
