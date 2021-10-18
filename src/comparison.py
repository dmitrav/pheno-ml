
import os, pandas, torch, numpy, random, umap, time
from hdbscan import HDBSCAN
from torch.nn import Sequential
from torchvision.io import read_image
from scipy.spatial.distance import pdist
from sklearn import metrics
from tqdm import tqdm

from src.self_supervised import DeepClassifier
from src.constants import cell_lines, drugs


def get_image_encodings_from_path(path, common_image_ids, transform, n=None, randomize=True):

    # get filenames to retrieve image ids
    if isinstance(common_image_ids, str):
        filenames = [f for f in os.listdir(path) if common_image_ids in f]

    elif isinstance(common_image_ids, list):
        # filenames must contain all ids
        filenames = [f for f in os.listdir(path) if sum([1 for x in common_image_ids if x in f]) == len(common_image_ids)]
    else:
        raise ValueError('Unknown common image ids: {}'.format(common_image_ids))

    if n is not None:
        if randomize:
            filenames = random.sample(filenames, n)
        else:
            filenames = sorted(filenames)[-n:]

    image_ids = {
        'filenames': filenames,
        'cell_lines': [f.split('_')[0] for f in filenames],
        'plates': [f.split('_')[2] for f in filenames],
        'wells': [f.split('_')[3] for f in filenames],
        'dates': ['_'.join(f.split('_')[-3:]) for f in filenames]
    }

    encodings = []
    # get encodings
    # print('processing images from {}'.format(path))
    for file in filenames:
        img = read_image(path + file)
        img_encoded = transform(img)
        encodings.append(img_encoded.detach().cpu().numpy())

    return encodings, image_ids


def get_f_transform(method_name, device=torch.device('cpu')):

    if method_name == 'resnet50':
        # upload pretrained resnet50
        transform = lambda x: x / 255.
    elif method_name == 'byol_renset50':
        # upload  resnet50, pretrained with byol
        transform = lambda x: x / 255.
    else:
        # upload my models
        path_to_model = '/Users/andreidm/ETH/projects/pheno-ml/res/models/{}/'.format(method_name)
        model = DeepClassifier().to(device)
        # load a trained deep classifier to use it in the transform
        model.load_state_dict(torch.load(path_to_model + 'best.torch', map_location=device))
        model.eval()
        # truncate to the layer with learned representations
        model = Sequential(*list(model.model.children())[:-4])
        # create a transform function with weakly supervised classifier
        transform = lambda x: model(torch.Tensor(numpy.expand_dims((x / 255.), axis=0)).to(device)).reshape(-1)

    return transform


def get_image_encodings_from_path(path, common_image_ids, transform, n=None, randomize=True):

    # get filenames to retrieve image ids
    if isinstance(common_image_ids, str):
        filenames = [f for f in os.listdir(path) if common_image_ids in f]

    elif isinstance(common_image_ids, list):
        # filenames must contain all ids
        filenames = [f for f in os.listdir(path) if sum([1 for x in common_image_ids if x in f]) == len(common_image_ids)]
    else:
        raise ValueError('Unknown common image ids: {}'.format(common_image_ids))

    if n is not None:
        if randomize:
            filenames = random.sample(filenames, n)
        else:
            filenames = sorted(filenames)[-n:]

    image_ids = {
        'filenames': filenames,
        'cell_lines': [f.split('_')[0] for f in filenames],
        'plates': [f.split('_')[2] for f in filenames],
        'wells': [f.split('_')[3] for f in filenames],
        'dates': ['_'.join(f.split('_')[-3:]) for f in filenames]
    }

    encodings = []
    # get encodings
    # print('processing images from {}'.format(path))
    for file in tqdm(filenames):
        img = read_image(path + file)
        img_encoded = transform(img)
        encodings.append(img_encoded.detach().cpu().numpy())

    return encodings, image_ids


def get_wells_of_drug_for_cell_line(cell_line, drug, plate=''):

    meta_path = '/Users/andreidm/ETH/projects/pheno-ml/data/metadata/'
    cell_plate_paths = [meta_path + file for file in os.listdir(meta_path) if cell_line in file and plate in file]

    wells = []
    for path in cell_plate_paths:
        data = pandas.read_csv(path)
        drug_data = data.loc[data['Drug'] == drug, ['Final_conc_uM', 'Well']]
        max_conc_wells = drug_data.loc[drug_data['Final_conc_uM'] == drug_data['Final_conc_uM'].max(), 'Well']
        wells.extend(list(max_conc_wells))
        if len(wells) > 0:
            plate = path.split('_')[-1].split('.')[0]
            return wells, plate


def calculate_similarity_of_pair(codes_A, codes_B):

    d_2 = []
    d_cos = []
    d_corr = []
    d_bray = []
    d_mahal = []

    for code_a in codes_A:
        for code_b in codes_B:

            d_2.append(pdist([code_a, code_b], metric='euclidean'))
            d_cos.append(pdist([code_a, code_b], metric='cosine'))
            d_corr.append(pdist([code_a, code_b], metric='correlation'))
            d_bray.append(pdist([code_a, code_b], metric='braycurtis'))
            d_mahal.append(pdist([code_a, code_b], metric='mahalanobis'))

    res = {
        'euclidean': numpy.median(d_2),
        'cosine': numpy.median(d_cos),
        'correlation': numpy.median(d_corr),
        'braycurtis': numpy.median(d_bray),
        'mahalanobis': numpy.median(d_mahal)

    }

    return res


def compare_similarity(path_to_data, methods):

    save_to = '/Users/andreidm/ETH/projects/pheno-ml/res/comparison/similarity/'
    if not os.path.exists(save_to):
        os.makedirs(save_to)

    results = {'group_by': [], 'method_name': [], 'comparison': [],
               'euclidean': [], 'cosine': [], 'correlation': [], 'braycurtis': [], 'mahalonobis': []}

    for method_name in methods:

        transform = get_f_transform(method_name, torch.device('cpu'))

        for cell_line in tqdm(cell_lines):

            mtx_wells, plate = get_wells_of_drug_for_cell_line(cell_line, 'Methotrexate')
            mtx_controls_wells, _ = get_wells_of_drug_for_cell_line(cell_line, 'DMSO', plate=plate)
            mtx_controls_wells = random.sample(mtx_controls_wells, len(mtx_wells))

            ptx_wells, plate = get_wells_of_drug_for_cell_line(cell_line, 'Pemetrexed')
            ptx_controls_wells, _ = get_wells_of_drug_for_cell_line(cell_line, 'DMSO', plate=plate)
            ptx_controls_wells = random.sample(ptx_controls_wells, len(ptx_wells))

            mtx_codes = []
            for well in mtx_wells:
                encodings, _ = get_image_encodings_from_path(path_to_data, [well, cell_line], transform, n=10, randomize=False)
                mtx_codes.extend(encodings)

            mtx_controls_codes = []
            for well in mtx_controls_wells:
                encodings, _ = get_image_encodings_from_path(path_to_data, [well, cell_line], transform, n=10, randomize=False)
                mtx_controls_codes.extend(encodings)

            ptx_codes = []
            for well in ptx_wells:
                encodings, _ = get_image_encodings_from_path(path_to_data, [well, cell_line], transform, n=10, randomize=False)
                ptx_codes.extend(encodings)

            ptx_controls_codes = []
            for well in ptx_controls_wells:
                encodings, _ = get_image_encodings_from_path(path_to_data, [well, cell_line], transform, n=10, randomize=False)
                ptx_controls_codes.extend(encodings)

            # compare Methotrexate with Pemetrexed
            results['group_by'].append(cell_line)
            results['method_name'].append(method_name)
            results['comparison'].append('MTX-PTX')
            comparison = calculate_similarity_of_pair(mtx_codes, ptx_codes)
            results['euclidean'].append(comparison['euclidean'])
            results['cosine'].append(comparison['cosine'])
            results['correlation'].append(comparison['correlation'])
            results['braycurtis'].append(comparison['braycurtis'])
            results['mahalanobis'].append(comparison['mahalanobis'])

            # compare Methotrexate with DMSO (control)
            results['group_by'].append(cell_line)
            results['method'].append(method_name)
            results['comparison'].append('MTX-DMSO')
            comparison = calculate_similarity_of_pair(mtx_codes, mtx_controls_codes)
            results['euclidean'].append(comparison['euclidean'])
            results['cosine'].append(comparison['cosine'])
            results['correlation'].append(comparison['correlation'])
            results['braycurtis'].append(comparison['braycurtis'])
            results['mahalanobis'].append(comparison['mahalanobis'])

            # compare Pemetrexed with DMSO (control)
            results['group_by'].append(cell_line)
            results['method'].append(method_name)
            results['comparison'].append('PTX-DMSO')
            comparison = calculate_similarity_of_pair(ptx_codes, ptx_controls_codes)
            results['euclidean'].append(comparison['euclidean'])
            results['cosine'].append(comparison['cosine'])
            results['correlation'].append(comparison['correlation'])
            results['braycurtis'].append(comparison['braycurtis'])
            results['mahalanobis'].append(comparison['mahalanobis'])

    results = pandas.DataFrame(results)
    results.to_csv(save_to + 'similarity.csv', index=False)


def collect_and_save_clustering_results_for_multiple_parameter_sets(path_to_images, methods, range_with_step, uid=''):
    """ Cluster the dataset over multiple parameters, evaluate results and save results as a dataframe. """

    save_to = '/Users/andreidm/ETH/projects/pheno-ml/res/comparison/clustering/'
    if not os.path.exists(save_to):
        os.makedirs(save_to)

    results = {'group_by': [], 'method_name': [], 'min_cluster_size': [],
               'n_clusters': [], 'noise': [], 'silhouette': [], 'calinski_harabasz': [], 'davies_bouldin': [],
               'consistency_cells': [], 'consistency_drugs': []}

    for method_name in methods:

        # TODO: set up correct transformations
        transform = get_f_transform(method_name)

        for cell_line in tqdm(cell_lines):

            all_codes = []
            all_ids = []
            for drug in drugs:
                if drug not in ['PBS', 'DMSO']:
                    max_conc_drug_wells, _ = get_wells_of_drug_for_cell_line(cell_line, drug)

                    drug_codes = []
                    drug_ids = []
                    for well in max_conc_drug_wells:
                        codes, ids = get_image_encodings_from_path(path_to_images, [well, cell_line], transform, n=10, randomize=False)
                        drug_codes.extend(codes)
                        drug_ids.extend(ids)

                    all_codes.extend(drug_codes)
                    all_ids.extend(drug_ids)

            # TODO: make sure the dims are correct
            encodings = numpy.array(all_codes)

            for min_cluster_size in tqdm(range(*range_with_step)):

                start = time.time()
                reducer = umap.UMAP(n_neighbors=min_cluster_size, metric='euclidean')
                embedding = reducer.fit_transform(encodings)
                # cluster encodings
                clusterer = HDBSCAN(metric='euclidean', min_samples=1, min_cluster_size=min_cluster_size, allow_single_cluster=False)
                clusterer.fit(embedding)
                clusters = clusterer.labels_
                print('umap + hdbscan clustering with n = {}, took {} s'.format(min_cluster_size, int(time.time() - start)))

                n_clusters = numpy.max(clusters) + 1
                noise = int(numpy.sum(clusters == -1) / len(clusters) * 100)

                try:
                    silhouette = metrics.silhouette_score(embedding, clusters)
                    calinski_harabasz = metrics.calinski_harabasz_score(embedding, clusters)
                    davies_bouldin = metrics.davies_bouldin_score(embedding, clusters)
                except ValueError:
                    # single cluster
                    silhouette, calinski_harabasz, davies_bouldin = -1, -1, -1

                consisten_c, consisten_d = calculate_clustering_consistency(clusters, image_ids)

                results['group_by'].append(cell_line)
                results['method'].append(model)
                results['setting'].append(setting)

                results['min_cluster_size'].append(min_cluster_size)
                results['n_clusters'].append(n_clusters)
                results['noise'].append(noise)
                results['silhouette'].append(silhouette)
                results['calinski_harabasz'].append(calinski_harabasz)
                results['davies_bouldin'].append(davies_bouldin)
                results['consistency_cells'].append(consisten_c)
                results['consistency_drugs'].append(consisten_d)

    results = pandas.DataFrame(results)
    results.to_csv(save_to + 'clustering_{}.csv'.format(uid), index=False)


if __name__ == "__main__":

    path_to_data = '/Users/andreidm/ETH/projects/pheno-ml/data/cropped/training/single_class/'
    # compare_similarity(path_to_data, ['resnet50'])

    collect_and_save_clustering_results_for_multiple_parameter_sets(path_to_data, ['resnet50'], (10, 160, 10),
                                                                    uid='by_cell_lines')