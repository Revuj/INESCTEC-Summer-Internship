import glob
import os

import cv2
import matplotlib.pyplot as plt
from keras.models import model_from_json
import pandas as pd
from keras import layers
from keras import models
from keras import regularizers
from keras.preprocessing.image import ImageDataGenerator


def predict_with_testset(model):
    map_characters = {
        0: 'Abdullah_II_of_Jordan',
        1: 'Aditya_Seal',
        2: 'Aishwarya_Rai',
        3: 'Alain_Traore',
        4: 'Alex_Gonzaga',
        5: 'Angelique_Kidjo',
        6: 'Anne_Princess_Royal',
        7: 'Aya_Miyama',
        8: 'Cavaco_Silva',
        9: 'Conan_O_Brian',
        10: 'Dalai_Lama',
        11: 'Zelia_Duncan'

    }


    path = '/home/revujenation/PycharmProjects/Estagio_INESCTEC/summer_internship_dataset/dataset/test'
    dataframe = pd.read_csv("/home/revujenation/PycharmProjects/Estagio_INESCTEC/summer_internship_dataset/test.csv")

    correct = 0
    incorrect = 0
    for index, row in dataframe.iterrows():
        image = cv2.imread(path + '/' + row['filename'])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width, depth = image.shape

        # Resize and normalize image to fit the model imput
        pic = cv2.resize(image, (100, 100)).astype('float32') / 255.
        predict_img = model.predict(pic.reshape(1, 100, 100, 3))


        # Get the prediction with higher confidence score
        predict = [round(pred, 5) * 100 for pred in predict_img[0]]
        idx = predict.index(max(predict))
        if map_characters[idx] == row['label']:
            correct += 1
        else:
            incorrect += 1


    print(str(correct/(correct+incorrect)))

def main():
    json_file = open('/home/revujenation/PycharmProjects/Estagio_INESCTEC/summer_internship_dataset/model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights('/home/revujenation/PycharmProjects/Estagio_INESCTEC/summer_internship_dataset/diff4_dataset_weights.h5', by_name=True)
    predict_with_testset(loaded_model)

if __name__ == '__main__':
    main()