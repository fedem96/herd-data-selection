from keras import Sequential
from keras.layers import Conv3D, Dropout, Flatten, Dense, regularizers, MaxPooling3D, BatchNormalization, Activation
from keras.optimizers import Adam


def get_model():

    model = Sequential()

    input_shape = (10, 10, 10, 1)

    model.add(
        Conv3D(filters=32, kernel_size=2, activation='relu', padding='same',
               kernel_regularizer=regularizers.l2(0.00001), input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(
        Conv3D(filters=32, kernel_size=2, padding='same',
               kernel_regularizer=regularizers.l2(0.00001)))
    model.add(Activation('relu'))
    model.add(
        Conv3D(filters=64, kernel_size=2, padding='same',
               kernel_regularizer=regularizers.l2(0.00001)))
    model.add(Activation('relu'))
    model.add(
        Conv3D(filters=64, kernel_size=2, padding='same',
               kernel_regularizer=regularizers.l2(0.00001)))
    model.add(Activation('relu'))

    model.add(MaxPooling3D(2))

    model.add(
        Conv3D(filters=128, kernel_size=2, padding='same',
               kernel_regularizer=regularizers.l2(0.00001)))
    model.add(Activation('relu'))
    model.add(
        Conv3D(filters=128, kernel_size=2, padding='same',
               kernel_regularizer=regularizers.l2(0.00001)))
    model.add(Activation('relu'))
    model.add(
        Conv3D(filters=256, kernel_size=2, padding='same',
               kernel_regularizer=regularizers.l2(0.00001)))
    model.add(Activation('relu'))
    model.add(
        Conv3D(filters=256, kernel_size=2, padding='same',
               kernel_regularizer=regularizers.l2(0.00001)))
    model.add(Activation('relu'))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(64, kernel_regularizer=regularizers.l2(0.01)))
    model.add(Activation('relu'))
    model.add(Dense(128, kernel_regularizer=regularizers.l2(0.01)))
    model.add(Activation('relu'))
    model.add(Dense(256, kernel_regularizer=regularizers.l2(0.01)))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, kernel_regularizer=regularizers.l2(0.01)))
    model.add(Activation('sigmoid'))

    opt = Adam(lr=0.001)

    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    model.summary()

    return model


def get_fit_settings():
    return {'batch_size': 128, 'epochs': 150}
