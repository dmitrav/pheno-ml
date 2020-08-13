
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


def get_closest_time_point_index(control_times, drug_time):
    """ Finds the time in control that's the closest to a given drug time,
        :returns an index of it """

    if drug_time <= numpy.min(control_times):
        # if (the first) drug time is earlier than (the first) control time,
        # return the index of the first control time
        return 0
    elif drug_time >= numpy.max(control_times):
        # if (the last) drug time is later than (the last) control time,
        # return the index of the last control time
        return control_times.shape[0]-1
    else:
        for i in range(control_times.shape[0]-1):
            # find the interval for a drug time
            if control_times[i] <= drug_time <= control_times[i+1]:
                # compare diffs to find the closest between left and right
                if (control_times[i] - drug_time) < (control_times[i+1] - drug_time):
                    return i
                else:
                    return i+1


def get_average_sample_encodings(path_to_encodings, sample_ids):
    """ This method is used to calculate average controls and drug concentration replicates. """

    samples = []
    for id in sample_ids:
        if os.path.exists(path_to_encodings + id + '.csv'):
            sample = pandas.read_csv(path_to_encodings + id + '.csv')
            samples.append(sample.values)
        else:
            print("control: well {} not found".format(id))

    average_sample = []
    for i in range(2, samples[0].shape[1]):
        feature = []
        for sample in samples:
            feature.append(sample[:, i])
        feature = numpy.mean(feature, axis=0)
        average_sample.append(feature)

    sample_encodings = numpy.array(average_sample).T
    time_scale = samples[0][:, 1]

    return sample_encodings, time_scale


if __name__ == "__main__":

    """  """

    meta_data = pandas.read_csv("/Volumes/biol_imsb_sauer_1/users/Mauro/Cell_culture_data/190310_LargeScreen/imageData/metadata/ACHN_CL3_P1.csv")
    path_to_encodings = "/Users/andreidm/ETH/projects/pheno-ml/data/batch_1/ACHN_CL3_P1/"

    control = 'DMSO'
    metric = 'euclidean'

    control_data = meta_data[(meta_data['Drug'] == control) & (meta_data['Final_conc_uM'] == 367.)]
    control_ids = control_data['Row'].astype('str') + control_data['Column'].astype('str')
    average_control, control_times = get_average_sample_encodings(path_to_encodings, control_ids)

    drugs_data = meta_data[meta_data['Drug'] != control]
    drug_names = drugs_data['Drug'].dropna().unique()

    for drug_name in tqdm(drug_names):
        print(drug_name, "is being processed")

        # add concentraions of this drug
        drug_data = drugs_data[drugs_data['Drug'] == drug_name]
        drug_cons = drug_data['Final_conc_uM'].dropna()
        drug_ids = drug_data['Row'].astype('str') + drug_data['Column'].astype('str')

        unique_cons = drug_cons.unique()

        pyplot.figure()
        all_dists = []
        cvs_time_axis = []
        for i in range(len(unique_cons)):

            print("data for c = {} is being processed".format(unique_cons[i]))

            # get ids of different concentrations
            con_ids = drug_ids.values[numpy.where(drug_cons == unique_cons[i])[0]]

            average_drug_con, drug_times = get_average_sample_encodings(path_to_encodings, con_ids)
            cvs_time_axis = drug_times[:]

            drug_con_to_control_dist = []
            for j in range(len(drug_times)):

                closest_time_point_in_control = get_closest_time_point_index(control_times, drug_times[j])
                control = average_control[closest_time_point_in_control, :]

                dist = pdist([average_drug_con[j, :], control], metric=metric)[0]
                drug_con_to_control_dist.append(dist)

            all_dists.append(drug_con_to_control_dist)

            pyplot.subplot(121)
            pyplot.plot(drug_times, drug_con_to_control_dist, label='c={}'.format(round(unique_cons[i], 3)), linewidth=1)
            pyplot.title("{}, {} distance to control".format(drug_name, metric))
            pyplot.legend()
            pyplot.grid()

        # calculate cvs
        cvs = []
        all_dists = numpy.array(all_dists)
        for i in range(all_dists.shape[1]):
            cv = numpy.std(all_dists[:, i].flatten()) / numpy.mean(all_dists[:, i].flatten())
            cvs.append(cv)

        # find max variation coef before drug injection
        injection_index = numpy.where(cvs_time_axis >= 0)[0][0]
        max_cv_before_injection = numpy.max(cvs[:injection_index])

        # find the moment when all the rest CV are bigger than max before injection
        time_onset = 0
        for i in range(injection_index, len(cvs)):
            if numpy.all(numpy.array(cvs[i:]) > max_cv_before_injection):
                time_onset = cvs_time_axis[i]
                break

        pyplot.subplot(122)
        pyplot.plot(cvs_time_axis, cvs, linewidth=1)
        pyplot.axvline(x=time_onset, c='b', label='onset time')
        pyplot.title("{}, variation coefficient in distances".format(drug_name))
        pyplot.legend()
        pyplot.grid()

        pyplot.show()
        # pyplot.savefig(
        #     "/Users/andreidm/ETH/projects/pheno-ml/res/img_distances_ACHN_CL3_P1/averaged/{}.pdf".format(drug_name))