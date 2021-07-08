from keras import Sequential
from keras.layers import Conv3D, Dropout, Flatten, Dense, regularizers, MaxPooling3D, AveragePooling3D
from keras.optimizers import Adam


def get_model():

    model = Sequential()

    input_shape = (10, 10, 10, 1)

    model.add(
        Conv3D(filters=16, kernel_size=3, activation='relu', padding='same',
               kernel_regularizer=regularizers.l2(0.00001), input_shape=input_shape))
    model.add(
        Conv3D(filters=32, kernel_size=3, activation='relu', padding='same',
               kernel_regularizer=regularizers.l2(0.00001)))
    model.add(
        Conv3D(filters=64, kernel_size=3, activation='relu', padding='same',
               kernel_regularizer=regularizers.l2(0.00001)))
    model.add(
        Conv3D(filters=128, kernel_size=3, activation='relu', padding='same',
               kernel_regularizer=regularizers.l2(0.00001)))
    model.add(MaxPooling3D(2))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
    model.add(Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid', kernel_regularizer=regularizers.l2(0.01)))

    opt = Adam(lr=0.0001)

    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    model.summary()

    return model


def get_fit_settings():
    return {'batch_size': 64, 'epochs': 150}
