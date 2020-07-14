
from keras.models import Model
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler, TensorBoard

if __name__ == "__main__":

    path = "/Volumes/biol_imsb_sauer_1/users/Andrei/cell_line_images/batch_1/"

    BATCH_SIZE = 256
    EPOCHS = 10

    # TODO: try bigger size, no rescaling, flow color_mode='grayscale'

    train_datagen = ImageDataGenerator(rescale=1./255)
    train_batches = train_datagen.flow_from_directory(path, target_size=(64, 64), shuffle=True, class_mode='input', batch_size=BATCH_SIZE)

    # ENCODER
    input_img = Input(shape=(64, 64, 3))
    x = Conv2D(48, (3, 3), activation='relu', padding='same')(input_img)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(96, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(192, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    encoded = Conv2D(32, (1, 1), activation='relu', padding='same')(x)

    # LATENT SPACE
    latentSize = (8, 8, 32)

    # DECODER
    direct_input = Input(shape=latentSize)
    x = Conv2D(192, (1, 1), activation='relu', padding='same')(direct_input)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(192, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(96, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(48, (3, 3), activation='relu', padding='same')(x)
    decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

    # COMPILE
    encoder = Model(input_img, encoded)
    decoder = Model(direct_input, decoded)
    autoencoder = Model(input_img, decoder(encoded))

    autoencoder.compile(optimizer='Adam', loss='binary_crossentropy')

    history = autoencoder.fit_generator(train_batches,
                                        steps_per_epoch=train_batches.samples // BATCH_SIZE,
                                        epochs=EPOCHS,
                                        verbose=2,
                                        callbacks=[TensorBoard(log_dir='/tmp/autoencoder')])



