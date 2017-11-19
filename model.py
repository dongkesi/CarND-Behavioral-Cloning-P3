import csv
# import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.models import Sequential, optimizers
from keras.layers import Flatten, Dense, Lambda, Conv2D, MaxPool2D, Cropping2D, Dropout
from keras.callbacks import EarlyStopping
import platform
import os

BATCH_SIZE = 256
EPOCHS = 1
IS_AUGMENT = True
AUG_MULTIPLY = 2
PATH = './h-data/IMG/'
csv_path = './h-data/driving_log.csv'

delimiter = '\\'

def augment(x, y):
    augmented_images, augmented_measurements = [], []
    for image, measurement in zip(x, y):
        augmented_images.append(np.fliplr(image))
        augmented_measurements.append(-measurement)

    return augmented_images, augmented_measurements


def generator(samples, file_path ='./data/IMG/', is_augment = True, batch=32):
    n_samples = len(samples)
    # print('n_samples', n_samples)
    while 1:
        samples = shuffle(samples)
        batch_size = int(batch / AUG_MULTIPLY) if is_augment else batch
        # print('batch_size', batch_size)
        for offset in range(0, n_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images = []
            angles = []
            for batch_sample in batch_samples:
                if is_augment:
                    camera = np.random.randint(3)
                    try:
                        image = plt.imread(file_path + batch_sample[camera].split(delimiter)[-1])
                    except PermissionError:
                        print(batch_sample[camera].split(delimiter))
                    images.append(image)
                    center_angle = float(batch_sample[3])
                    correction = 0.18
                    if camera == 0:
                        angle = center_angle
                    elif camera == 1:
                        angle = center_angle + correction
                    else:
                        angle = center_angle - correction
                    angles.append(angle)

                    # center_image = plt.imread(file_path + batch_sample[0].split(delimiter)[-1])
                    # left_image = plt.imread(file_path + batch_sample[1].split(delimiter)[-1])
                    # right_image = plt.imread(file_path + batch_sample[2].split(delimiter)[-1])
                    #
                    # correction = 0.2
                    # center_angle = float(batch_sample[3])
                    # left_angle = center_angle + correction
                    # right_angle = center_angle - correction
                    #
                    # images.extend([center_image, left_image, right_image])
                    # angles.extend([center_angle, left_angle, right_angle])

                else:
                    name = file_path + batch_sample[0].split(delimiter)[-1]
                    center_image = plt.imread(name)
                    center_angle = float(batch_sample[3])
                    images.append(center_image)
                    angles.append(center_angle)

            if is_augment:
                aug_images, aug_angles = augment(images, angles)
                X_train = np.concatenate((np.array(images), np.array(aug_images)), axis=0)
                y_train = np.concatenate((np.array(angles), np.array(aug_angles)), axis=0)
            else:
                X_train = np.array(images)
                y_train = np.array(angles)

            yield shuffle(X_train, y_train)


def get_samples(csv_path):
    _samples = []
    with open(csv_path) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            _samples.append(line)
    return _samples[1:]


def net():
    model = Sequential()
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160, 320, 3)))
    # (160, 320, 3) -> (65, 320, 3)
    model.add(Cropping2D(cropping=((70, 25), (0, 0))))
    # (65, 320, 3) -> (31, 158, 24)
    model.add(Conv2D(24, 5, strides=(2, 2), padding='valid', activation='elu'))
    # model.add(Dropout(0.5))
    # (31, 158, 24) -> (14, 77, 36)
    # model.add(Conv2D(36, 5, strides=(2, 2), padding='valid', activation='elu'))
    model.add(MaxPool2D((2, 2), strides=(2, 2)))
    # model.add(Dropout(0.5))
    # (14, 77, 36) -> (5, 37, 48)
    model.add(Conv2D(48, 5, strides=(2, 2), padding='valid', activation='elu'))
    # model.add(Dropout(0.5))
    # (5, 37, 48) -> (3, 35, 64)
    # model.add(Conv2D(64, 3, strides=(1, 1), padding='valid', activation='elu'))
    model.add(MaxPool2D((2, 2), strides=(2, 2)))
    # model.add(Dropout(0.5))
    # (3, 35, 64) -> (1, 33, 64)
    model.add(Conv2D(64, 3, strides=(1, 1), padding='valid', activation='elu'))
    model.add(Dropout(0.5))
    #
    model.add(Flatten())
    model.add(Dense(100, activation='elu'))
    model.add(Dropout(0.5))
    model.add(Dense(50, activation='elu'))
    model.add(Dense(10, activation='elu'))
    model.add(Dense(1))

    return model


def train(model, train_data, validation_data, batch_size, epochs):

    train_generator = generator(train_data, PATH, is_augment=IS_AUGMENT, batch=batch_size)
    validation_generator = generator(validation_data, PATH, is_augment=False, batch=batch_size)

    n_train_samples = len(train_data) * AUG_MULTIPLY if IS_AUGMENT else len(train_data)
    n_train_steps = int(np.ceil(n_train_samples / float(batch_size)))

    n_valid_samples = len(validation_data)
    n_valid_steps = int(np.ceil(n_valid_samples / float(batch_size)))
    print(n_train_samples, n_train_steps, n_valid_samples, n_valid_steps)

    cbks = [EarlyStopping(patience=2)]
    weights_path = 'my_model_weights.h5'
    if os.path.isfile(weights_path):
        print('load weights')
        model.load_weights(weights_path)
    model.compile(loss='mse', optimizer=optimizers.Adam(lr=5e-04))
    history = model.fit_generator(generator=train_generator,
                                  steps_per_epoch=n_train_steps,
                                  validation_data=validation_generator,
                                  validation_steps=n_valid_steps,
                                  epochs=epochs,
                                  callbacks=cbks,
                                  workers=1)
    print('save weights and model ')
    model.save_weights(weights_path)
    model.save('model.h5')
    return history


origin_samples = get_samples(csv_path)
train_samples, validation_samples = train_test_split(shuffle(origin_samples), test_size=0.2)
model_obj = net()
history_obj = train(model_obj, train_samples, validation_samples, BATCH_SIZE, EPOCHS)


print(history_obj.history.keys())
### plot the training and validation loss for each epoch
plt.plot(history_obj.history['loss'])
plt.plot(history_obj.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()



