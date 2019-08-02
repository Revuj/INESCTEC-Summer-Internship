import matplotlib.pyplot as plt
import pandas as pd
from keras import layers, applications, Model, Input
from keras import models
from keras import regularizers
from keras.applications import ResNet50, VGG16
from keras.layers import MaxPool2D, Dropout, Dense, Flatten
from keras.preprocessing.image import ImageDataGenerator


def smooth_curve(points, factor=0.8):
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return smoothed_points


def main():
    train_data = ImageDataGenerator(rescale=1. / 255,
                                    rotation_range=30,
                                    horizontal_flip=True
                                    )

    val_data = ImageDataGenerator(rescale=1. / 255)

    train_diff = pd.read_csv('train_diff0.csv')
    val_diff = pd.read_csv('val_diff0.csv')

    train_gen = train_data.flow_from_dataframe(
        dataframe=train_diff,
        directory='dataset/diff',
        target_size=(100, 100),
        batch_size=16,
        x_col="filename", y_col="label", class_mode="categorical")

    validation_gen = val_data.flow_from_dataframe(
        dataframe=val_diff,
        directory='dataset/diff',
        target_size=(100, 100),
        batch_size=16,
        x_col="filename", y_col="label", class_mode="categorical")

    train_gen = train_data.flow_from_dataframe(
        dataframe = train_diff,
        directory = 'diff',
        target_size=(100, 100),
        batch_size=16,
        x_col="filepath", y_col="label", class_mode="categorical")

    validation_gen = val_data.flow_from_dataframe(
            dataframe = val_diff,
            directory = 'diff',
            target_size=(100, 100),
            batch_size=16,
            x_col="filepath", y_col="label", class_mode="categorical")

    model = models.Sequential()
    model.add(layers.Conv2D(8, (3, 3), activation='relu', input_shape=(100, 100, 3)))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(16, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='final_conv'))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Flatten())
    model.add(layers.Dense(64, kernel_regularizer=regularizers.l2(0.01), activation='relu'))

    model.add(layers.Dropout(rate=0.5))

    model.add(layers.Dense(12, kernel_regularizer=regularizers.l2(0.01), activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

    # serialize model to JSON
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)

    history = model.fit_generator(
        train_gen,
        steps_per_epoch=238,
        epochs=30,
        validation_data=validation_gen,
        validation_steps=60)

    # Save weights
    model.save_weights('diff0_dataset_weights_resnet50.h5')

    # Plot history
    acc = smooth_curve(history.history['acc'])
    val_acc = smooth_curve(history.history['val_acc'])
    loss = smooth_curve(history.history['loss'])
    val_loss = smooth_curve(history.history['val_loss'])
    epochs = range(1, len(acc) + 1)

    fig, axs = plt.subplots(1, 2, figsize=(15, 7))
    axs[0].plot(epochs, acc, 'bo', label='Training_acc')
    axs[0].plot(epochs, val_acc, 'b', label='Validation_acc')
    axs[0].legend()
    axs[1].plot(epochs, loss, 'bo', label='Training_loss')
    axs[1].plot(epochs, val_loss, 'b', label='Validation_loss')
    axs[1].legend()
    plt.show()


if __name__ == '__main__':
    main()
