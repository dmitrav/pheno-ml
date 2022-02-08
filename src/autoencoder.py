
import numpy, pandas, os
from matplotlib import pyplot
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import torchvision, torch
from pl_bolts.models.self_supervised import SwAV
from torchvision.transforms import Resize, Grayscale, ToPILImage, ToTensor


def create_autoencoder_model(target_size=(128, 128)):

    if target_size[0] == 64:
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
        latentSize = (8, 8, 16)

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

    elif target_size[0] == 128:

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
    else:
        raise ValueError("Model not specified")

    # COMPILE
    encoder = Model(input_img, encoded)
    decoder = Model(direct_input, decoded)
    ae = Model(input_img, decoder(encoded))

    return encoder, decoder, ae


def visualize_reconstruction(data_batches, trained_ae):

    x_batch = next(data_batches)[0]

    pyplot.figure()
    for i in range(0, 7):
        pyplot.subplot(2, 7, i + 1)
        pyplot.imshow(x_batch[i][:, :, 0], cmap='gray')
        pyplot.title("original")

    for i in range(0, 7):
        pyplot.subplot(2, 7, i + 8)
        input_img = numpy.expand_dims(x_batch[i], axis=0)
        reconstructed_img = trained_ae(input_img)
        pyplot.imshow(reconstructed_img[0][:, :, 0], cmap='gray')
        pyplot.title("reconstruction")

    # pyplot.tight_layout()
    pyplot.show()


def visualize_results(data_batches, trained_ae, save_to):

    encoder = Model(trained_ae.input, trained_ae.layers[-2].output)
    x_batch = next(data_batches)[0]

    pyplot.figure(figsize=(15,7))
    for i in range(0, 7):
        pyplot.subplot(3, 7, i + 1)
        pyplot.imshow(x_batch[i][:, :, 0], cmap='gray')
        pyplot.title("original")

    for i in range(0, 7):
        pyplot.subplot(3, 7, i + 8)
        input_img = numpy.expand_dims(x_batch[i], axis=0)
        encoded_img = encoder(input_img)

        # reduce dimensions from 3 to 2
        new_shape = (int(numpy.sqrt(numpy.product(encoded_img[0].shape))), int(numpy.sqrt(numpy.product(encoded_img[0].shape))))
        encoded_img = numpy.reshape(encoded_img[0], new_shape)

        pyplot.imshow(encoded_img, cmap='gray')
        pyplot.title("encodings")

    for i in range(0, 7):
        pyplot.subplot(3, 7, i + 15)
        input_img = numpy.expand_dims(x_batch[i], axis=0)
        reconstructed_img = trained_ae(input_img)
        pyplot.imshow(reconstructed_img[0][:, :, 0], cmap='gray')
        pyplot.title("reconstruction")

    pyplot.tight_layout()
    pyplot.show()
    # pyplot.savefig(save_to+'reconstruction.pdf')


def plot_loss(history, n_epochs):

    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(n_epochs)
    pyplot.figure()
    pyplot.plot(epochs, loss, 'bo', label='Training loss')
    pyplot.plot(epochs, val_loss, 'b', label='Validation loss')
    pyplot.title('Training and validation loss')
    pyplot.legend()
    pyplot.grid()
    pyplot.show()


def train_autoencoder(path="/Users/andreidm/ETH/projects/pheno-ml/data/squeezed/training/", model_name='ae'):

    BATCH_SIZE = 32
    EPOCHS = 4

    target_size = (128, 128)

    train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.1)

    train_batches = train_datagen.flow_from_directory(path, target_size=target_size, color_mode='grayscale',
                                                      subset='training',
                                                      shuffle=True, class_mode='input', batch_size=BATCH_SIZE)

    val_batches = train_datagen.flow_from_directory(path, target_size=target_size, color_mode='grayscale',
                                                    subset='validation',
                                                    shuffle=True, class_mode='input', batch_size=BATCH_SIZE)

    encoder, decoder, autoencoder = create_autoencoder_model(target_size)
    autoencoder.compile(optimizer=Adam(learning_rate=0.0003), loss='binary_crossentropy')
    autoencoder.summary()

    if True:
        # if pretrained
        print("loading weights")
        epoch_to_start_from = 14
        latest = '/Users/andreidm/ETH/projects/pheno-ml/res/weights/ae_cropped_{}_at_{}.h5'.format(target_size[0], epoch_to_start_from)
        autoencoder.load_weights(latest)

        print("checking current encodings")
        visualize_reconstruction(val_batches, autoencoder)

        print("fitting further...")

    history = autoencoder.fit(train_batches,
                              steps_per_epoch=train_batches.samples // BATCH_SIZE,
                              # steps_per_epoch=10000,
                              epochs=EPOCHS,
                              verbose=1,
                              validation_data=val_batches,
                              validation_steps=val_batches.samples // BATCH_SIZE,
                              # validation_steps=10000,
                              callbacks=[ModelCheckpoint("../res/weights/{}_{}".format(model_name, target_size[0]) + "_at_14+{epoch}.h5")])

    plot_loss(history, EPOCHS)
    visualize_reconstruction(val_batches, autoencoder)


def create_and_save_encodings_for_well(well_meta, image_paths, save_to_path, single_drug=None, method=None):

    encodings = []

    if method is None:
        _, _, autoencoder = create_autoencoder_model()
        # autoencoder.load_weights('/Users/andreidm/ETH/projects/pheno-ml/res/weights/ae128_at_21_0.6800.h5')
        autoencoder.load_weights('/Users/andreidm/ETH/projects/pheno-ml/res/weights/ae_cropped_128_adam_at_16_0.6880.h5')
        encoder = Model(autoencoder.input, autoencoder.layers[-2].output)

        for path in image_paths:

            image = tf.io.read_file(path)
            image = tf.image.decode_jpeg(image, channels=1)
            image = tf.image.resize(image, [autoencoder.input.shape[1], autoencoder.input.shape[2]])
            image /= 255.
            image = numpy.expand_dims(image, axis=0)

            encoded_image = encoder(image)
            encoded_image = numpy.array(encoded_image).flatten()
            encodings.append(encoded_image)
    else:

        weight_path = 'https://pl-bolts-weights.s3.us-east-2.amazonaws.com/swav/swav_imagenet/swav_imagenet.pth.tar'
        model = SwAV.load_from_checkpoint(weight_path, strict=True)
        model.freeze()

        transform = lambda x: model(
            torch.unsqueeze(  # add batch dimension
                ToTensor()(  # convert PIL to tensor
                    Grayscale(num_output_channels=3)(  # apply grayscale, keeping 3 channels
                        ToPILImage()(  # conver to PIL to apply grayscale
                            Resize(size=224)(  # and resnet is trained with 224
                                Resize(size=128)(x)  # images are 256, but all models are trained with 128
                            )
                        )
                    )
                ), 0)
        ).reshape(-1).detach().cpu().numpy()

        for path in image_paths:

            img = torchvision.io.read_image(path)
            img_encoded = transform(img.to('cpu'))
            encodings.append(img_encoded.flatten())

    well_features = pandas.DataFrame(encodings)
    well_features.insert(0, 'Time', well_meta['Time'].values)
    well_features.insert(0, 'Date', well_meta['Time.gmt'].values)

    well_features.to_csv(save_to_path + well_meta['Well'].values[0] + '_{}_{}.csv'.format(single_drug, method), index=False)
    print(well_meta['Well'].values[0] + ' encoded and saved')


def generate_encodings_for_batches(batch_range, single_drug=None, method=None):

    all_meta_data = pandas.read_csv("/Users/andreidm/ETH/projects/pheno-ml/data/pheno-ml-metadata.csv")

    path_to_images = '/Users/andreidm/ETH/projects/pheno-ml/data/cropped/batch_{}/'
    path_to_plate_meta_data = '/Users/andreidm/ETH/projects/pheno-ml/data/metadata/{}.csv'

    for batch in batch_range:

        print("\nprocessing batch {}...\n".format(batch))

        for cl_plate in os.listdir(path_to_images.format(batch)):
             if not cl_plate.startswith('.'):
                save_to = path_to_images.format(batch) + cl_plate + '/'
                print("\nprocessing folder {}...\n".format(cl_plate))

                plate_meta_data = pandas.read_csv(path_to_plate_meta_data.format(cl_plate))

                if single_drug:
                    drug_wells = plate_meta_data.loc[plate_meta_data['Drug'] == single_drug, 'Well'].unique()
                    control_wells = plate_meta_data.loc[(plate_meta_data['Drug'] == 'DMSO') & (plate_meta_data['Final_conc_uM'] == 367.), 'Well'].unique()
                    unique_wells = numpy.concatenate([drug_wells, control_wells])
                else:
                    unique_wells = plate_meta_data.loc[~pandas.isnull(plate_meta_data['Drug']), 'Well'].unique()

                cell_line, _, plate = cl_plate.replace('.csv', '').split('_')

                for well in unique_wells:

                    # align images with meta data to substitute dates with time axis
                    meta_data = all_meta_data.loc[(all_meta_data['cell'] == cell_line) &
                                                  (all_meta_data['Well'] == well) &
                                                  (all_meta_data['source_plate'] == plate), :]

                    # sort both by dates
                    image_names = sorted([path for path in os.listdir(path_to_images.format(batch) + cl_plate) if path.endswith('.jpg') and path.split('_')[3] == well])
                    meta_data.sort_values('Time.gmt')  # just in case

                    image_paths = [path_to_images.format(batch) + cl_plate + '/' + name for name in image_names]

                    if meta_data.shape[0] == 0 or len(image_paths) == 0:
                        print('no images or meta data found for well {} -> skipping'.format(well))
                        continue
                    else:

                        try:
                            assert meta_data.shape[0] == len(image_paths)
                        except AssertionError:
                            print("{}, {}: meta and images can't be mapped -> filtering \'Conf\'".format(cl_plate, well))
                            meta_data = meta_data.loc[~pandas.isnull(meta_data['Conf']), :]

                            assert meta_data.shape[0] == len(image_paths)

                        create_and_save_encodings_for_well(meta_data, image_paths, save_to, single_drug=single_drug, method=method)


def get_trained_autoencoder():

    target_size = (128, 128)
    weights = '/Users/andreidm/ETH/projects/pheno-ml/res/weights/ae_cropped_128_adam_at_16_0.6880.h5'
    _, _, autoencoder = create_autoencoder_model(target_size)
    autoencoder.compile()
    autoencoder.summary()
    autoencoder.load_weights(weights)

    return autoencoder


def load_model_and_plot_results(save_to):

    path = "/Users/andreidm/ETH/projects/pheno-ml/data/cropped/training/"
    weights = '/Users/andreidm/ETH/projects/pheno-ml/res/weights/ae_cropped_128_adam_at_16_0.6880.h5'

    BATCH_SIZE = 32
    target_size = (128, 128)

    train_datagen = ImageDataGenerator(rescale=1. / 255, validation_split=0.1)
    train_batches = train_datagen.flow_from_directory(path, target_size=target_size, color_mode='grayscale', subset='training', shuffle=True, class_mode='input', batch_size=BATCH_SIZE)
    val_batches = train_datagen.flow_from_directory(path, target_size=target_size, color_mode='grayscale', subset='validation', shuffle=True, class_mode='input', batch_size=BATCH_SIZE)

    _, _, autoencoder = create_autoencoder_model(target_size)
    autoencoder.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy')
    autoencoder.summary()

    print("loading weights")
    autoencoder.load_weights(weights)

    print("plotting images, encodings and reconstructions")
    visualize_results(val_batches, autoencoder, save_to)


if __name__ == "__main__":

    if False:
        path = '/Users/andreidm/ETH/projects/pheno-ml/data/cropped/training/'
        model_name = 'ae_cropped'
        train_autoencoder(path=path, model_name=model_name)

    if False:
        save_to = '/Users/andreidm/Library/Mobile Documents/com~apple~CloudDocs/ETHZ/papers_posters/pheno-ml/ICML21/img/'
        load_model_and_plot_results(save_to)

    if True:
        generate_encodings_for_batches([1], single_drug='Cladribine', method='swav_resnet50')



