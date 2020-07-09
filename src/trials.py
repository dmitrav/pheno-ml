
import tensorflow as tf
import numpy, scipy, os, pandas, time
from tqdm import tqdm
from matplotlib import pyplot
from scipy.spatial.distance import pdist


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
    pyplot.savefig("/Users/andreidm/ETH/projects/pheno-ml/res/distances_ACHN_CL3_P1/averaged/{}.pdf".format(drug_name))


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

                # read a drug replicate
                drug_rep_image = tf.keras.preprocessing.image.load_img(path + batch + cell_line + drug_image_paths[con_ids[k]][j], target_size=target_size)
                drug_rep_features = get_image_features(model, drug_rep_image)

                # read a control
                control_image = tf.keras.preprocessing.image.load_img(path + batch + cell_line + control_image_paths[random_control_ids[0]][j], target_size=target_size)
                control_features = get_image_features(model, control_image)

                dist = pdist([drug_rep_features.tolist(), control_features.tolist()], metric='correlation')[0]
                one_replicate_distances.append(dist)

            pyplot.plot([one_replicate_distances.index(x) for x in one_replicate_distances], one_replicate_distances,
                        label=con_ids[k], linewidth=1)

        pyplot.title("{}, c={},\ncorrelation distance to control".format(drug_name, round(unique_cons[i], 3)))
        pyplot.legend()
        pyplot.grid()
        # pyplot.show()
        pyplot.savefig("/Users/andreidm/ETH/projects/pheno-ml/res/distances_ACHN_CL3_P1/single/{}_c={}.pdf".format(drug_name, round(unique_cons[i], 3)))


if __name__ == "__main__":

    if False:
        plot_correlation_distance_for_a_pair()

    if True:

        meta_data = pandas.read_csv("/Volumes/biol_imsb_sauer_1/users/Mauro/Cell_culture_data/190310_LargeScreen/imageData/metadata/ACHN_CL3_P1.csv")

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

            print()
