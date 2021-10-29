
import numpy
from matplotlib import pyplot
from torchvision.io import read_image
from torchvision.transforms import Resize
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D


def create_autoencoder_model(target_size=(128, 128)):

    # ENCODER
    input_img = Input(shape=(*target_size, 1))
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(input_img)
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
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

    # COMPILE
    encoder = Model(input_img, encoded)
    decoder = Model(direct_input, decoded)
    ae = Model(input_img, decoder(encoded))

    return encoder, decoder, ae


def get_trained_autoencoder(weights='/Users/andreidm/ETH/projects/pheno-ml/res/weights/ae_cropped_128_adam_at_16_0.6880.h5',
                            target_size=(128, 128)):

    _, _, autoencoder = create_autoencoder_model(target_size)
    autoencoder.compile()
    autoencoder.summary()
    autoencoder.load_weights(weights)

    return autoencoder


if __name__ == "__main__":

    weights = '/Users/andreidm/ETH/projects/pheno-ml/res/weights/ae_cropped_128_adam_at_16_0.6880.h5'
    image_size = (128, 128)
    # create model and load weights
    autoencoder = get_trained_autoencoder(weights=weights, target_size=image_size)

    # beak the model into parts
    encoder = Model(autoencoder.input, autoencoder.layers[-2].output)
    decoder = Model(autoencoder.layers[-1].input, autoencoder.layers[-1].layers[-1].output)

    # take an example image
    example_path = '/Users/andreidm/ETH/projects/pheno-ml/data/cropped/batch_2/MDAMB231_CL3_P2/MDAMB231_CL3_P2_A7_1_2019y09m27d_22h23m.jpg'
    img = read_image(example_path)

    # preprocess the data: scale, resize, expand dims for model input
    preprocessor = lambda x: numpy.expand_dims(Resize(size=image_size[0])(x / 255.).numpy(), axis=-1)

    # get encodings
    codes = encoder(preprocessor(img))

    # manipulate image encodings
    flat_codes = numpy.reshape(codes, -1)
    # introduce some gaussian noise (for example)
    perturbed_codes = flat_codes + numpy.random.normal(numpy.mean(flat_codes) / 10, numpy.std(flat_codes) / 10, flat_codes.shape)
    stacked_codes = numpy.reshape(perturbed_codes, (1, 16, 16, 16))

    # get reconstruction
    rec = decoder(stacked_codes)

    # plot initial image
    pyplot.figure()
    pyplot.imshow(numpy.reshape(preprocessor(img), image_size), cmap='gray')
    pyplot.title("initial image")
    # plot reconstructed image
    pyplot.figure()
    pyplot.imshow(numpy.reshape(rec, image_size), cmap='gray')
    pyplot.title("reconstructed image")

    pyplot.show()


