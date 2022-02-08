
import tensorflow as tf
import numpy, scipy, os, pandas, time, json
from tqdm import tqdm
from matplotlib import pyplot
from scipy.spatial.distance import pdist
from scipy.signal import savgol_filter
from multiprocessing import Process, Pool


def get_image_features(model, image):

    img_data = tf.keras.preprocessing.image.img_to_array(image)
    img_data = numpy.expand_dims(img_data, axis=0)

    assert model._name == 'resnet50'
    img_data = tf.keras.applications.resnet50.preprocess_input(img_data)

    features_3d = model.predict(img_data)
    # print("resnet features dim:", features_3d.shape)
    features_averaged_1d = features_3d[0].mean(axis=0).mean(axis=0)

    return features_averaged_1d


def plot_correlation_distance_for_a_pair():

    target_size = (224, 224)
    model = tf.keras.applications.ResNet50(include_top=False, weights='imagenet')

    path = "/Volumes/biol_imsb_sauer_1/users/Mauro/Cell_culture_data/190310_LargeScreen/imageData/batch_1/ACHN_CL3_P1/"

    control = []
    cell_line_a = []
    cell_line_b = []
    for file in os.listdir(path):

        if file.startswith("ACHN_CL3_P1_A3"):
            control.append(path + file)
        elif file.startswith("ACHN_CL3_P1_A5"):
            cell_line_a.append(path + file)
        elif file.startswith("ACHN_CL3_P1_A4"):
            cell_line_b.append(path + file)
        else:
            pass

    control = sorted(control)
    cell_line_a = sorted(cell_line_a)
    cell_line_b = sorted(cell_line_b)

    a_c_distances = []
    b_c_distances = []
    for i in range(len(control)):
        image_c = tf.keras.preprocessing.image.load_img(control[i], target_size=target_size)
        image_a = tf.keras.preprocessing.image.load_img(cell_line_a[i], target_size=target_size)
        image_b = tf.keras.preprocessing.image.load_img(cell_line_b[i], target_size=target_size)

        features_c = get_image_features(model, image_c)
        features_a = get_image_features(model, image_a)
        features_b = get_image_features(model, image_b)

        a_to_c_distance = pdist([features_a.tolist(), features_c.tolist()], metric='correlation')
        b_to_c_distance = pdist([features_b.tolist(), features_c.tolist()], metric='correlation')

        # print(a_to_b_distance)

        a_c_distances.append(a_to_c_distance[0])
        b_c_distances.append(b_to_c_distance[0])

    pyplot.plot([a_c_distances.index(x) for x in a_c_distances], a_c_distances, label='A5')
    pyplot.plot([b_c_distances.index(x) for x in b_c_distances], b_c_distances, label='A4')

    pyplot.title("treated cell lines vs control")
    pyplot.legend()
    pyplot.grid()
    pyplot.show()


def plot_correlation_distance_for_averaged_samples(drug_name, drug_info, control_ids):

    path = "/Volumes/biol_imsb_sauer_1/users/Mauro/Cell_culture_data/190310_LargeScreen/imageData/"

    batch = 'batch_1/'
    cell_line = 'ACHN_CL3_P1/'

    target_size = (224, 224)
    model = tf.keras.applications.ResNet50(include_top=False, weights='imagenet')

    unique_cons = drug_info['cons'].unique()

    control_ids = control_ids.values
    random_control_subset = numpy.random.choice(control_ids, 4, replace=False).tolist()  # pick 4 controls to compare to

    pyplot.figure()
    for i in tqdm(range(len(unique_cons))):

        print("data for c = {} is being processed".format(unique_cons[i]))

        # get ids of different concentrations
        con_ids = drug_info['ids'].values[numpy.where(drug_info['cons'] == unique_cons[i])[0]]

        drug_image_paths = {}
        # get images of a drug for different replicates (of some concentration)
        for j in range(len(con_ids)):

            drug_image_paths[con_ids[j]] = []
            for file in os.listdir(path + batch + cell_line):

                if file.startswith(cell_line.replace('/', '_') + con_ids[j]):
                    drug_image_paths[con_ids[j]].append(file)

            drug_image_paths[con_ids[j]] = sorted(drug_image_paths[con_ids[j]])

        control_image_paths = {}
        # get images of control for different replicates
        for j in range(len(control_ids)):

            control_image_paths[control_ids[j]] = []
            for file in os.listdir(path + batch + cell_line):

                if file.startswith(cell_line.replace('/', '_') + control_ids[j]):
                    control_image_paths[control_ids[j]].append(file)

            control_image_paths[control_ids[j]] = sorted(control_image_paths[control_ids[j]])

        # TODO: check that there's equal number of pictures in each replicate
        #       otherwise -- equalize

        drug_dist_over_time = []
        for j in range(len(drug_image_paths[con_ids[0]])):  # paths of images

            drug_img_reps = []
            for k in range(len(con_ids)):  # concentration ids
                # read replicates
                rep = tf.keras.preprocessing.image.load_img(path + batch + cell_line + drug_image_paths[con_ids[k]][j], target_size=target_size)
                rep_features = get_image_features(model, rep)
                drug_img_reps.append(rep_features)

            drug_features = numpy.mean(numpy.vstack(drug_img_reps), axis=0)

            control_img_reps = []
            for id in random_control_subset:  # control ids
                # read replicates
                rep = tf.keras.preprocessing.image.load_img(path + batch + cell_line + control_image_paths[id][j], target_size=target_size)
                rep_features = get_image_features(model, rep)
                control_img_reps.append(rep_features)

            control_features = numpy.mean(numpy.vstack(control_img_reps), axis=0)

            drug_control_dist = pdist([drug_features.tolist(), control_features.tolist()], metric='correlation')
            drug_dist_over_time.append(drug_control_dist)

        pyplot.plot([drug_dist_over_time.index(x) for x in drug_dist_over_time], drug_dist_over_time,
                    label='c={}'.format(round(unique_cons[i], 3)), linewidth=1)

    pyplot.title("{}, correlation distance to control".format(drug_name))
    pyplot.legend()
    pyplot.grid()
    # pyplot.show()
    pyplot.savefig("/Users/andreidm/ETH/projects/pheno-ml/res/img_distances_ACHN_CL3_P1/averaged/{}.pdf".format(drug_name))


def plot_correlation_distance_for_single_samples(drug_name, drug_info, control_ids):

    path = "/Volumes/biol_imsb_sauer_1/users/Mauro/Cell_culture_data/190310_LargeScreen/imageData/"

    batch = 'batch_1/'
    cell_line = 'ACHN_CL3_P1/'

    target_size = (224, 224)
    model = tf.keras.applications.ResNet50(include_top=False, weights='imagenet')

    unique_cons = drug_info['cons'].unique()

    control_ids = control_ids.values
    random_control_ids = numpy.random.choice(control_ids, 1, replace=False)  # pick 1 control to compare to

    for i in tqdm(range(len(unique_cons))):

        pyplot.figure()
        print("data for c = {} is being processed".format(unique_cons[i]))

        # get ids of different concentrations
        con_ids = drug_info['ids'].values[numpy.where(drug_info['cons'] == unique_cons[i])[0]]

        drug_image_paths = {}
        # get images of a drug for different replicates (of some concentration)
        for j in range(len(con_ids)):

            drug_image_paths[con_ids[j]] = []
            for file in os.listdir(path + batch + cell_line):

                if file.startswith(cell_line.replace('/', '_') + con_ids[j]):
                    drug_image_paths[con_ids[j]].append(file)

            drug_image_paths[con_ids[j]] = sorted(drug_image_paths[con_ids[j]])

        control_image_paths = {}
        # get images of control for different replicates
        for j in range(len(random_control_ids)):

            control_image_paths[random_control_ids[j]] = []
            for file in os.listdir(path + batch + cell_line):

                if file.startswith(cell_line.replace('/', '_') + random_control_ids[j]):
                    control_image_paths[random_control_ids[j]].append(file)

            control_image_paths[random_control_ids[j]] = sorted(control_image_paths[random_control_ids[j]])

        # TODO: check that there's equal number of pictures in each replicate
        #       otherwise -- equalize

        for k in range(len(con_ids)):  # ids of replicates of certain concentration

            one_replicate_distances = []
            for j in range(len(drug_image_paths[con_ids[k]])):  # paths of images

                # print("i={}, k={}, j={}".format(i, k, j))

                # read a drug replicate
                drug_rep_image = tf.keras.preprocessing.image.load_img(path + batch + cell_line + drug_image_paths[con_ids[k]][j], target_size=target_size)
                drug_rep_features = get_image_features(model, drug_rep_image)

                # read a control
                control_image = tf.keras.preprocessing.image.load_img(path + batch + cell_line + control_image_paths[random_control_ids[0]][j], target_size=target_size)
                control_features = get_image_features(model, control_image)

                dist = pdist([drug_rep_features.tolist(), control_features.tolist()], metric='correlation')[0]
                one_replicate_distances.append(dist)

            print("plot called")
            pyplot.plot([one_replicate_distances.index(x) for x in one_replicate_distances], one_replicate_distances,
                        label=con_ids[k], linewidth=1)

        pyplot.title("{}, c={},\ncorrelation distance to control".format(drug_name, round(unique_cons[i], 3)))
        pyplot.legend()
        pyplot.grid()
        # pyplot.show()
        pyplot.savefig("/Users/andreidm/ETH/projects/pheno-ml/res/img_distances_ACHN_CL3_P1/single/{}_c={}.pdf".format(drug_name, round(unique_cons[i], 3)))
        print("plot saved")


def generate_snippets_of_data(drug_info, control_ids):
    """ This is to generate some examples of distance over time values for Mauro. """

    path = "/Volumes/biol_imsb_sauer_1/users/Mauro/Cell_culture_data/190310_LargeScreen/imageData/"

    batch = 'batch_1/'
    cell_line = 'ACHN_CL3_P1/'

    target_size = (224, 224)
    model = tf.keras.applications.ResNet50(include_top=False, weights='imagenet')

    unique_cons = drug_info['cons'].unique()

    control_ids = control_ids.values
    random_control_ids = numpy.random.choice(control_ids, 1, replace=False)  # pick 1 control to compare to

    snippets = {}

    for i in tqdm(range(len(unique_cons))):

        snippets[unique_cons[i]] = {}

        print("data for c = {} is being processed".format(unique_cons[i]))

        # get ids of different concentrations
        con_ids = drug_info['ids'].values[numpy.where(drug_info['cons'] == unique_cons[i])[0]]

        drug_image_paths = {}
        # get images of a drug for different replicates (of some concentration)
        for j in range(len(con_ids)):

            drug_image_paths[con_ids[j]] = []
            for file in os.listdir(path + batch + cell_line):

                if file.startswith(cell_line.replace('/', '_') + con_ids[j]):
                    drug_image_paths[con_ids[j]].append(file)

            drug_image_paths[con_ids[j]] = sorted(drug_image_paths[con_ids[j]])

        control_image_paths = {}
        # get images of control for different replicates
        for j in range(len(random_control_ids)):

            control_image_paths[random_control_ids[j]] = []
            for file in os.listdir(path + batch + cell_line):

                if file.startswith(cell_line.replace('/', '_') + random_control_ids[j]):
                    control_image_paths[random_control_ids[j]].append(file)

            control_image_paths[random_control_ids[j]] = sorted(control_image_paths[random_control_ids[j]])

        # TODO: check that there's equal number of pictures in each replicate
        #       otherwise -- equalize

        for k in range(len(con_ids)):  # ids of replicates of certain concentration

            one_replicate_distances = []
            for j in range(len(drug_image_paths[con_ids[k]])):  # paths of images

                # read a drug replicate
                drug_rep_image = tf.keras.preprocessing.image.load_img(path + batch + cell_line + drug_image_paths[con_ids[k]][j], target_size=target_size)
                drug_rep_features = get_image_features(model, drug_rep_image)

                # read a control
                control_image = tf.keras.preprocessing.image.load_img(path + batch + cell_line + control_image_paths[random_control_ids[0]][j], target_size=target_size)
                control_features = get_image_features(model, control_image)

                dist = pdist([drug_rep_features.tolist(), control_features.tolist()], metric='correlation')[0]
                one_replicate_distances.append(float(dist))

            snippets[unique_cons[i]][con_ids[k]] = one_replicate_distances

    return snippets


def plot_distances_for_cell_line_using_resnet():

    meta_data = pandas.read_csv(
        "/Volumes/biol_imsb_sauer_1/users/Mauro/Cell_culture_data/190310_LargeScreen/imageData/metadata/ACHN_CL3_P1.csv")

    control = 'DMSO'

    control_data = meta_data[(meta_data['Drug'] == control) & (meta_data['Final_conc_uM'] == 367.)]
    control_ids = control_data['Row'].astype('str') + control_data['Column'].astype('str')

    drugs_data = meta_data[meta_data['Drug'] != control]
    drug_names = drugs_data['Drug'].dropna().unique()

    drug_info = {}
    for drug_name in tqdm(drug_names):
        print(drug_name, "is being processed")

        # create a dict for this drug
        drug_info[drug_name] = {}
        # add concentraions of this drug
        drug_data = drugs_data[drugs_data['Drug'] == drug_name]
        drug_info[drug_name]['cons'] = drug_data['Final_conc_uM'].dropna()
        # add ids of this drug
        drug_ids = drug_data['Row'].astype('str') + drug_data['Column'].astype('str')
        drug_info[drug_name]['ids'] = drug_ids

        # plot_correlation_distance_for_single_samples(drug_name, drug_info[drug_name], control_ids)
        plot_correlation_distance_for_averaged_samples(drug_name, drug_info[drug_name], control_ids)


def generate_snippets_of_distances_for_cell_line_using_resnet():

    meta_data = pandas.read_csv(
        "/Volumes/biol_imsb_sauer_1/users/Mauro/Cell_culture_data/190310_LargeScreen/imageData/metadata/ACHN_CL3_P1.csv")

    control = 'DMSO'

    control_data = meta_data[(meta_data['Drug'] == control) & (meta_data['Final_conc_uM'] == 367.)]
    control_ids = control_data['Row'].astype('str') + control_data['Column'].astype('str')

    drugs_data = meta_data[meta_data['Drug'] != control]
    drug_names = drugs_data['Drug'].dropna().unique()

    results = {}
    drug_info = {}
    for drug_name in tqdm(drug_names):
        print(drug_name, "is being processed")

        # create a dict for this drug
        drug_info[drug_name] = {}
        # add concentraions of this drug
        drug_data = drugs_data[drugs_data['Drug'] == drug_name]
        drug_info[drug_name]['cons'] = drug_data['Final_conc_uM'].dropna()
        # add ids of this drug
        drug_ids = drug_data['Row'].astype('str') + drug_data['Column'].astype('str')
        drug_info[drug_name]['ids'] = drug_ids

        results[drug_name] = generate_snippets_of_data(drug_info[drug_name], control_ids)
        print(results[drug_name])

    with open("/Users/andreidm/ETH/projects/pheno-ml/res/distances_ACHN_CL3_P1/single/ACHN_CL3_P1.txt", 'w') as file:
        file.write(results.__str__())

    with open("/Users/andreidm/ETH/projects/pheno-ml/res/distances_ACHN_CL3_P1/single/ACHN_CL3_P1.json", 'w') as file:
        json.dump(results, file)


def plot_distances_single_controls_vs_averaged_one():
    """ Using already encodings, not pretrained net features. """

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


def get_average_swav_encodings(path_to_encodings, sample_ids, sample_name):
    """ Analog for Cladribine case """

    samples = []
    for id in sample_ids:
        if os.path.exists(path_to_encodings + id + '_Cladribine_swav_resnet50.csv'):
            sample = pandas.read_csv(path_to_encodings + id + '.csv')
            samples.append(sample.values)
        else:
            print("{}: well {} not found".format(sample_name, id))

    shapes = numpy.array([sample.shape[0] for sample in samples])
    if len(set(shapes)) > 1:
        # filter out samples of different shape
        most_freq_shape = numpy.argmax(numpy.bincount(shapes))
        samples = [sample for sample in samples if sample.shape[0] == most_freq_shape]
        print(numpy.sum(shapes == most_freq_shape), 'of', shapes.shape[0], '{} samples have shape {}, using only those'.format(sample_name, most_freq_shape))

    # compute average sample now
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


def get_average_sample_encodings(path_to_encodings, sample_ids, sample_name):
    """ This method is used to calculate average controls and drug concentration replicates. """

    samples = []
    for id in sample_ids:
        if os.path.exists(path_to_encodings + id + '.csv'):
            sample = pandas.read_csv(path_to_encodings + id + '.csv')
            samples.append(sample.values)
        else:
            print("{}: well {} not found".format(sample_name, id))

    shapes = numpy.array([sample.shape[0] for sample in samples])
    if len(set(shapes)) > 1:
        # filter out samples of different shape
        most_freq_shape = numpy.argmax(numpy.bincount(shapes))
        samples = [sample for sample in samples if sample.shape[0] == most_freq_shape]
        print(numpy.sum(shapes == most_freq_shape), 'of', shapes.shape[0], '{} samples have shape {}, using only those'.format(sample_name, most_freq_shape))

    # compute average sample now
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


def calculate_distances_for_batch_and_save_results(batch_number, metric='euclidean', control_name='DMSO', save_distances=True, save_dist_plots=True, save_cv_plots=False,
                                                   path_to_save_to='/Users/andreidm/ETH/projects/pheno-ml/res/distances/cropped/'):

    path_to_all_meta = "/Users/andreidm/ETH/projects/pheno-ml/data/metadata/"  # folder name, e.g. ACHN_CL3_P1
    path_to_batches = "/Users/andreidm/ETH/projects/pheno-ml/data/cropped/"

    print("\nbatch {} is being processed".format(batch_number))
    path_to_batch = path_to_batches + "batch_{}/".format(batch_number)

    for cell_line_folder in os.listdir(path_to_batch):
        print("\nfolder {} is being processed\n".format(cell_line_folder))
        if cell_line_folder.startswith("."):
            continue
        else:
            path_to_encodings = path_to_batch + cell_line_folder + '/'
            path_to_meta = path_to_all_meta + "{}.csv".format(cell_line_folder)

            meta_data = pandas.read_csv(path_to_meta)

            control_data = meta_data[(meta_data['Drug'] == control_name) & (meta_data['Final_conc_uM'] == 367.)]
            control_ids = control_data['Row'].astype('str') + control_data['Column'].astype('str')
            # average_control, control_times = get_average_sample_encodings(path_to_encodings, control_ids, 'control')
            average_control, control_times = get_average_swav_encodings(path_to_encodings, control_ids, 'control')

            drugs_data = meta_data[meta_data['Drug'] != control_name]
            # drug_names = drugs_data['Drug'].dropna().unique()
            drug_names = ['Cladribine']

            for drug_name in drug_names:
                print('\n', drug_name, "is being processed")

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

                    # average_drug_con, drug_times = get_average_sample_encodings(path_to_encodings, con_ids, '{}, (c={})'.format(drug_name,unique_cons[i]))
                    average_drug_con, drug_times = get_average_swav_encodings(path_to_encodings, con_ids, '{}, (c={})'.format(drug_name,unique_cons[i]))
                    cvs_time_axis = drug_times[:]

                    drug_con_to_control_dist = []
                    for j in range(len(drug_times)):
                        closest_time_point_in_control = get_closest_time_point_index(control_times, drug_times[j])
                        control_of_the_same_time = average_control[closest_time_point_in_control, :]

                        dist = pdist([average_drug_con[j, :], control_of_the_same_time], metric=metric)[0]
                        drug_con_to_control_dist.append(dist)

                    all_dists.append(drug_con_to_control_dist)

                    pyplot.plot(drug_times, drug_con_to_control_dist, label='c={}'.format(round(unique_cons[i], 3)),linewidth=1)
                    pyplot.title("{}, {} distance to control".format(drug_name, metric))
                    pyplot.legend()
                    pyplot.grid()

                if save_distances:

                    drug_result = {'time': drug_times.tolist()}
                    for i in range(len(unique_cons)):
                        con_key = str(round(unique_cons[i], 4))
                        drug_result[con_key] = all_dists[i]

                    one_time_path_for_mauro = '/Volumes/biol_imsb_sauer_1/users/Mauro/from_Andrei/swav_distances_batch_1/'
                    current_path = one_time_path_for_mauro + 'batch_{}/'.format(batch_number) + '{}/'.format(cell_line_folder)
                    if not os.path.exists(current_path):
                        os.makedirs(current_path)
                    with open(current_path + "/{}_{}.json".format(drug_name, metric), 'w') as file:
                        json.dump(drug_result, file)

                    current_path = path_to_save_to + 'batch_{}/'.format(batch_number) + '{}/'.format(cell_line_folder)
                    if not os.path.exists(current_path):
                        os.makedirs(current_path)
                    with open(current_path + "/{}_{}.json".format(drug_name, metric), 'w') as file:
                        json.dump(drug_result, file)

                    print('\ndistances for drug {} saved'.format(drug_name))

                if save_dist_plots:
                    current_path = path_to_save_to + 'batch_{}/'.format(batch_number) + '{}/'.format(cell_line_folder)
                    if not os.path.exists(current_path):
                        os.makedirs(current_path)

                    pyplot.savefig(current_path + "/{}_{}.pdf".format(drug_name, metric))
                    # pyplot.show()

                    print('\ndistance plots for drug {} saved'.format(drug_name))

                if save_cv_plots:
                    # calculate cvs
                    cvs = []
                    all_dists = numpy.array(all_dists)
                    for i in range(all_dists.shape[1]):
                        cv = numpy.std(all_dists[:, i].flatten()) / numpy.mean(all_dists[:, i].flatten())
                        cvs.append(cv)

                    window = len(cvs_time_axis) // 2 if (len(cvs_time_axis) // 2) % 2 > 0 else 1 + len(cvs_time_axis) // 2
                    cvs_smoothed = savgol_filter(cvs, window_length=window, polyorder=10)

                    # find max variation coef before drug injection
                    injection_index = numpy.where(cvs_time_axis >= 0)[0][0]
                    max_cv_before_injection = numpy.max(cvs_smoothed[:injection_index])

                    # find the moment when all the rest CV are bigger than max before injection
                    time_onset = 0
                    for i in range(injection_index, len(cvs_smoothed)):
                        if numpy.all(numpy.array(cvs_smoothed[i:]) > max_cv_before_injection):
                            time_onset = cvs_time_axis[i]
                            break

                    pyplot.figure()
                    pyplot.plot(cvs_time_axis, cvs, linewidth=1, label='cv raw')
                    pyplot.plot(cvs_time_axis, cvs_smoothed, linewidth=1, label='cv smoothed')
                    pyplot.axvline(x=time_onset, c='b', label='onset time')
                    pyplot.title("{}, variation coefficient in distances".format(drug_name))
                    pyplot.legend()
                    pyplot.grid()

                    current_path = path_to_save_to + 'batch_{}/'.format(batch_number) + '{}/'.format(cell_line_folder)
                    if not os.path.exists(current_path):
                        os.makedirs(current_path)

                    # pyplot.show()
                    pyplot.savefig(current_path + "/CV_{}_{}.pdf".format(drug_name, metric))
                    print('\ncv plots for drug {} saved'.format(drug_name))

                pyplot.close('all')


if __name__ == "__main__":

    if False:
        """ plot just one example of distance between control and drug,
            using resnet-50 features (quick and dirty, proof-of-principle) """
        plot_correlation_distance_for_a_pair()

        """ plot all distances for one cell line,
            using resnet-50 features (quick and dirty, proof-of-principle):
            - for single replicates,
            - or for single drug concentrations with averaged replicates """
        plot_distances_for_cell_line_using_resnet()

        """ generate snippets of data for Mauro to test T_onset calculation,
            using resnet-50 features (quick and dirty, proof-of-principle) """
        generate_snippets_of_distances_for_cell_line_using_resnet()

    if False:
        """ explore how single controls of one cell line & plate converge with time with the averaged control,
            using proper image encodings """
        plot_distances_single_controls_vs_averaged_one()

    if False:
        """ run main distance analysis and save results """

        save_distances = True
        save_dist_plots = True
        save_cv_plots = False
        path_to_save_to = '/Users/andreidm/ETH/projects/pheno-ml/res/swav_distances/'

        batches = [1]

        for batch in batches:
            calculate_distances_for_batch_and_save_results(batch,
                                                           metric='euclidean',
                                                           save_distances=save_distances,
                                                           save_dist_plots=save_dist_plots,
                                                           save_cv_plots=save_cv_plots,
                                                           path_to_save_to=path_to_save_to)

        for batch in batches:
            calculate_distances_for_batch_and_save_results(batch,
                                                           metric='braycurtis',
                                                           save_distances=save_distances,
                                                           save_dist_plots=save_dist_plots,
                                                           save_cv_plots=save_cv_plots,
                                                           path_to_save_to=path_to_save_to)

