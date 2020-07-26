
import numpy
from matplotlib import pyplot
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator


if __name__ == "__main__":

    path = "/Users/andreidm/ETH/projects/pheno-ml/data/training/"

    BATCH_SIZE = 32
    EPOCHS = 20

    target_size = (128, 128)

    train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.1)

    train_batches = train_datagen.flow_from_directory(path, target_size=target_size, color_mode='grayscale',
                                                      subset='training',
                                                      shuffle=True, class_mode='input', batch_size=BATCH_SIZE)

    val_batches = train_datagen.flow_from_directory(path, target_size=target_size, color_mode='grayscale',
                                                    subset='validation',
                                                    shuffle=True, class_mode='input', batch_size=BATCH_SIZE)

    # trial #1: 20 epochs -> val_loss: 0.6893
    # trial #2: 15 epochs -> val_loss: 0.6887 (reconstruction not very detailed)
    # trial #3: 5 epochs -> val_loss: 0.6884 (reconstruction more detailed)
    # trial #4: 10 epochs -> val_loss: 0.6850

    # ENCODER
    input_img = Input(shape=(*target_size, 1))
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(input_img)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(96, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    encoded = Conv2D(16, (1, 1), activation='relu', padding='same')(x)

    # LATENT SPACE
    latentSize = (16, 16, 16)

    # DECODER
    direct_input = Input(shape=latentSize)
    x = Conv2D(16, (1, 1), activation='relu', padding='same')(direct_input)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(96, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

    # COMPILE
    encoder = Model(input_img, encoded)
    decoder = Model(direct_input, decoded)
    autoencoder = Model(input_img, decoder(encoded))

    autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy')
    autoencoder.summary()

    if False:
        # if pretrained
        print("loading weights")
        latest = '/Users/andreidm/ETH/projects/pheno-ml/res/weights/ae64_at_30.h5'
        autoencoder.load_weights(latest)

        print("checking current encodings")
        x_batch = next(val_batches)[0]

        pyplot.figure()
        for i in range(0, 10):
            pyplot.subplot(2, 10, i + 1)
            pyplot.imshow(x_batch[i][:, :, 0], cmap='gray')
            pyplot.title("original")

        for i in range(0, 10):
            pyplot.subplot(2, 10, i + 11)
            input_img = numpy.expand_dims(x_batch[i], axis=0)
            reconstructed_img = autoencoder(input_img)
            pyplot.imshow(reconstructed_img[0][:, :, 0], cmap='gray')
            pyplot.title("reconstruction")

        pyplot.tight_layout()
        pyplot.show()

        print("fitting further...")

    history = autoencoder.fit(train_batches,
                              # steps_per_epoch=train_batches.samples // BATCH_SIZE,
                              steps_per_epoch=10000,
                              epochs=EPOCHS,
                              verbose=1,
                              validation_data=val_batches,
                              validation_steps=val_batches.samples // BATCH_SIZE,
                              # validation_steps=10000,
                              callbacks=[ModelCheckpoint("../res/weights/ae128_at_{epoch}.h5")])

    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(EPOCHS)
    pyplot.figure()
    pyplot.plot(epochs, loss, 'bo', label='Training loss')
    pyplot.plot(epochs, val_loss, 'b', label='Validation loss')
    pyplot.title('Training and validation loss')
    pyplot.legend()
    pyplot.grid()
    pyplot.show()

    x_batch = next(val_batches)[0]

    pyplot.figure()
    for i in range(0, 10):
        pyplot.subplot(2, 10, i + 1)
        pyplot.imshow(x_batch[i][:, :, 0], cmap='gray')
        pyplot.title("original")

    for i in range(0, 10):
        pyplot.subplot(2, 10, i + 11)
        input_img = numpy.expand_dims(x_batch[i], axis=0)
        reconstructed_img = autoencoder(input_img)
        pyplot.imshow(reconstructed_img[0][:, :, 0], cmap='gray')
        pyplot.title("reconstruction")

    pyplot.tight_layout()
    pyplot.show()


