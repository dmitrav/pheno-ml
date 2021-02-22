
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

from datetime import datetime

if __name__ == "__main__":

    heatmap = pandas.DataFrame(0, columns=['c1', 'c2', 'c3'], index=['r1', 'r2', 'r3'])

    print(heatmap)
