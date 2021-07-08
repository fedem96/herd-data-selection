'''import tensorflow as tf
from keras.backend.tensorflow_backend import set_session'''

from keras import Sequential
from keras.callbacks import EarlyStopping
from keras.layers import Conv2D, Dropout, Flatten, Dense, regularizers, MaxPooling2D, Activation, Reshape

from keras.optimizers import Adam


def get_model():

    model = Sequential()

    model.add(Reshape(input_shape=(20, 20, 20, 1), target_shape=(20, 20, 20)))

    model.add(
        Conv2D(filters=16, kernel_size=3, activation='relu', padding='same',
               kernel_regularizer=regularizers.l2(0.00001)))
    model.add(
        Conv2D(filters=32, kernel_size=3, activation='relu', padding='same',
               kernel_regularizer=regularizers.l2(0.00001)))
    model.add(
        Conv2D(filters=64, kernel_size=3, activation='relu', padding='same',
               kernel_regularizer=regularizers.l2(0.00001)))
    model.add(MaxPooling2D(2))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
    model.add(Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid', kernel_regularizer=regularizers.l2(0.01)))

    opt = Adam(lr=0.001)

    # es = EarlyStopping(monitor='val_loss',
    #                    min_delta=0,
    #                    patience=2,
    #                    verbose=0, mode='auto')

    # reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
    #                               patience=5, min_lr=0.001)

    # filepath = "weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
    # checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])#, callbacks=es)
    model.summary()

    '''config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.05
    set_session(tf.Session(config=config))'''

    return model
