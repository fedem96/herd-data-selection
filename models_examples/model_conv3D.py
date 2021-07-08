""" modello che utilizza le reti convoluzionali; i tensori sono trattati come oggetti a tre dimensioni spaziali e singolo canale """

from keras import Sequential
from keras.layers import Conv3D, Dropout, Flatten, Dense, regularizers, MaxPooling3D
from keras.optimizers import Adam


class ModelConv3D(Sequential):
    def __init__(self):
        super(ModelConv3D, self).__init__()

        input_shape = (20, 20, 20, 1)

        self.add(
            Conv3D(filters=16, kernel_size=3, strides=2, activation='relu', input_shape=input_shape, padding='same',
                   kernel_regularizer=regularizers.l2(0.00001)))
        self.add(
            Conv3D(filters=32, kernel_size=3, activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(0.00001)))
        self.add(
            Conv3D(filters=64, kernel_size=3, activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(0.00001)))
        self.add(MaxPooling3D(2))
        self.add(Dropout(0.25))

        self.add(Flatten())
        self.add(Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
        self.add(Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
        self.add(Dropout(0.5))
        self.add(Dense(1, activation='sigmoid', kernel_regularizer=regularizers.l2(0.01)))

        opt = Adam(lr=0.001)
        self.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
        self.summary()
