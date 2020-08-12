
import numpy, pandas, os
from tqdm import tqdm
from matplotlib import pyplot
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from src import autoencoder as ae
from matplotlib import pyplot
from scipy.spatial.distance import pdist


if __name__ == "__main__":

    """  """

    meta_data = pandas.read_csv("/Volumes/biol_imsb_sauer_1/users/Mauro/Cell_culture_data/190310_LargeScreen/imageData/metadata/ACHN_CL3_P1.csv")

    control = 'DMSO'

    control_data = meta_data[(meta_data['Drug'] == control) & (meta_data['Final_conc_uM'] == 367.)]
    control_ids = control_data['Row'].astype('str') + control_data['Column'].astype('str')

    path_to_encodings = "/Users/andreidm/ETH/projects/pheno-ml/data/batch_1/ACHN_CL3_P1/"

    controls = []
    for id in control_ids.values:
        if os.path.exists(path_to_encodings + id + '.csv'):
            control = pandas.read_csv(path_to_encodings + id + '.csv')
            controls.append(control.values)
        else:
            print("well {} not found".format(id))

    average_control = []
    for i in range(2, controls[0].shape[1]):
        feature = []
        for control in controls:
            feature.append(control[:, i])
        feature = numpy.mean(feature, axis=0)
        average_control.append(feature)

    average_control = numpy.array(average_control).T

    pyplot.figure()
    for k in range(len(controls)):

        distance_to_average = []
        for i in range(average_control.shape[0]):

            single_control_feature = controls[k][i, 2:].tolist()
            average_control_feature = average_control[i, :].tolist()
            dist = pdist([single_control_feature, average_control_feature], metric='correlation')[0]
            distance_to_average.append(dist)

        pyplot.plot(controls[k][:, 1], distance_to_average, label=control_ids.values[k])

    pyplot.grid()
    pyplot.tight_layout()
    pyplot.show()



    pass