
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
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from scipy.spatial.distance import pdist
from scipy.signal import savgol_filter
from src import constants

from src import cluster_analysis

if __name__ == "__main__":

    # meta = pandas.read_csv("/Users/andreidm/ETH/projects/pheno-ml/data/pheno-ml-metadata.csv")

    x = numpy.linspace(0, 1, 30)
    z = x
    y = x ** 2 + 1

    for cmap in ['flag', 'prism', 'ocean', 'gist_earth', 'terrain', 'gist_stern',
            'gnuplot', 'gnuplot2', 'CMRmap', 'cubehelix', 'brg',
            'gist_rainbow', 'rainbow', 'jet', 'nipy_spectral', 'gist_ncar']:


        fig = pyplot.figure()
        ax = fig.add_subplot(111, projection='3d')

        scatter = ax.scatter(x, y, z, c=[x for x in range(len(x))], cmap=cmap)

        ax.legend(labels=[str(x) for x in x])
        ax.grid()

        pyplot.title(cmap)
        pyplot.show()

    print()

