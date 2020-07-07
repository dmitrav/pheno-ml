
import tensorflow as tf
import numpy, tqdm, scipy, os
from matplotlib import pyplot
from scipy.spatial.distance import pdist


def get_image_features(image):

    img_data = tf.keras.preprocessing.image.img_to_array(image)
    img_data = numpy.expand_dims(img_data, axis=0)

    img_data = tf.keras.applications.resnet50.preprocess_input(img_data)

    features_3d = model.predict(img_data)

    # print("resnet features dim:", features_3d.shape)

    features_averaged_1d = features_3d[0].mean(axis=0).mean(axis=0)

    return features_averaged_1d


if __name__ == "__main__":

    model = tf.keras.applications.ResNet50(include_top=False, weights='imagenet')

    target_size = (224, 224)

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

        features_c = get_image_features(image_c)
        features_a = get_image_features(image_a)
        features_b = get_image_features(image_b)

        a_to_c_distance = pdist([features_a.tolist(), features_c.tolist()], metric='correlation')
        b_to_c_distance = pdist([features_b.tolist(), features_c.tolist()], metric='correlation')

        # print(a_to_b_distance)

        a_c_distances.append(a_to_c_distance[0])
        b_c_distances.append(b_to_c_distance[0])

    pyplot.plot([a_c_distances.index(x) for x in a_c_distances], a_c_distances, color='green')
    pyplot.plot([b_c_distances.index(x) for x in b_c_distances], b_c_distances, color='red')

    pyplot.title("treated cell lines vs control")
    pyplot.grid()
    pyplot.show()