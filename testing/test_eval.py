import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.models import model_from_json
from keras.utils import to_categorical
from sklearn.metrics import confusion_matrix


def data_loader(path, pic_size):
    """Loads the test images.
    """
    map_characters = {
        'Abdullah_II_of_Jordan':0,
        'Aditya_Seal':1,
        'Aishwarya_Rai':2,
        'Alain_Traore':3,
        'Alex_Gonzaga':4,
        'Angelique_Kidjo':5,
        'Anne_Princess_Royal':6,
        'Aya_Miyama':7,
        'Cavaco_Silva':8,
        'Conan_O_Brian':9,
        'Dalai_Lama':10,
        'Zelia_Duncan':11
    }

    x, y = [], []
    dataframe = pd.read_csv("/home/revujenation/PycharmProjects/Estagio_INESCTEC/summer_internship_dataset/test.csv")

    for index, row in dataframe.iterrows():
        temp = cv2.imread(path + '/' + row['filename'])
        try:
            temp = cv2.resize(temp, (pic_size, pic_size)).astype('float32') / 255.
            b, g, r = cv2.split(temp)
            temp = cv2.merge((r, g, b))
            x.append(temp)
            y.append(map_characters[row['label']])
        except Exception as e:
            print(e)
            print(row['filename'])

    n_classes = len(set(y))
    x = np.array(x)
    y = np.array(y)
    print(y)
    print(n_classes)
    y = to_categorical(y, n_classes)
    print(y)
    return x, y


def evaluate(json, weights, x, y):
    """Having the model and it's weights, the model is tested, returning the accuracy and loss.
    """
    print(json, weights)
    model_path = 'model_experiment_xxx.json'
    weights_path = 'weights_experiment_xxx.h5'

    json_file = open(json, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)

    model.load_weights(weights)

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    model.summary()


    test_loss, test_acc = model.evaluate(x, y)
    test_pred = model.predict(x)

    print('Test loss: ', test_loss)
    print('Test acc: ', test_acc)

    normalize = False

    classes = ['Aishwarya_Rai', 'Conan_O_Brian', 'Abdullah_II_of_Jordan', 'Cavaco_Silva',
		'Aditya_Seal', 'Anne_Princess_Royal', 'Alex_Gonzaga', 'Dalai_Lama', 'Angelique_Kidjo',
		'Zelia_Duncan', 'Aya_Miyama', 'Alain_Traore']

    cm = confusion_matrix(y.argmax(axis=1), test_pred.argmax(axis=1))

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(ylabel='True label', xlabel='Predicted label',
           xticks=np.arange(cm.shape[1]), yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes)

    plt.setp(ax.get_xticklabels(), rotation=45,
             ha='right', rotation_mode='anchor')

    thresh = cm.max() / 2.
    if normalize:
        form_str = '.2f'
    else:
        form_str = 'd'
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], form_str),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    # fig.tight_layout()
    fig.set_figheight(10)
    fig.set_figwidth(10)
    plt.show()

    return test_acc, test_loss


def main():
    path = '/home/revujenation/PycharmProjects/Estagio_INESCTEC/summer_internship_dataset/dataset/test'
    x, y = data_loader(path, 100)
    evaluate('model.json', '175_random_1_dataset_weights.h5', x, y)





if __name__ == '__main__':
    main()
