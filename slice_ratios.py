import pandas as pd
import os


def main():
    files_names = os.listdir("normalized_ratios")
    for filename in files_names:
        print(filename)
        dataframe = pd.read_csv("normalized_ratios/" + filename)
        new_ratios = dataframe[['ex-en,en-en',
                                'ls-sto,sto-li',
                                'cl_lips',
                                'ca_lips',
                                'cb_lips',
                                'cl_eyes',
                                'ca_eyes',
                                'cb_eyes',
                                'cl_eyebrows',
                                'ca_eyebrows',
                                'cb_eyebrows']]
        new_ratios.to_csv("new_normalized_ratios/" + filename)


if __name__ == '__main__':
    main()