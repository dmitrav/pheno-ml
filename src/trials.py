
import numpy
from matplotlib import pyplot
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from src import autoencoder as ae


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

    encoder, decoder, autoencoder = ae.create_autoencoder_model(target_size)
    autoencoder.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy')
    autoencoder.summary()

    print("loading weights")
    weights = '/Users/andreidm/ETH/projects/pheno-ml/res/weights/ae128_at_21_0.6800.h5'
    autoencoder.load_weights(weights)

    print("checking current encodings")
    ae.visualize_results(val_batches, autoencoder)