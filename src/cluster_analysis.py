
import numpy, pandas, os, umap, seaborn, time
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot
from src import constants


def get_well_drug_mapping_for_cell_line(cell_line_folder, path_to_meta='/Volumes/biol_imsb_sauer_1/users/Mauro/Cell_culture_data/190310_LargeScreen/imageData/metadata/'):
    """ Retrieve a well-to-drug mapping for a cell line folder. """

    cell_line_meta = pandas.read_csv(path_to_meta + cell_line_folder + '.csv')
    wells = cell_line_meta['Well'].values
    drugs = cell_line_meta['Drug'].values
    del cell_line_meta

    mapping = {}
    for i in range(wells.shape[0]):
        mapping[wells[i]] = drugs[i]

    return mapping


def collect_encodings_of_cell_line_by_time_points(cell_line_name, time_point='zero', path_to_batches="/Users/andreidm/ETH/projects/pheno-ml/data/"):

    drug_names = []
    cell_line_encodings = []

    for batch in range(1, 8):
        path_to_batch = path_to_batches + "batch_{}/".format(batch)
        for folder in os.listdir(path_to_batch):
            if cell_line_name in folder:

                well_drug_map = get_well_drug_mapping_for_cell_line(folder)

                path_to_encodings = path_to_batch + folder + '/'

                # take only every second path to speed up calculations
                well_paths = [file for file in os.listdir(path_to_encodings) if file.endswith('.csv')]
                well_names = [well_drug_map[file.replace('.csv', '')] for file in os.listdir(path_to_encodings) if file.endswith('.csv')]

                for i in range(len(well_paths)):

                    well_encodings = pandas.read_csv(path_to_encodings + well_paths[i]).values
                    time = well_encodings[:, 1].astype('float32')
                    encodings = well_encodings[:, 2:].astype('float32')
                    del well_encodings

                    if time_point == 'zero':
                        # get number of encodings before drug administration
                        n_times_before_drug = time[time < 0].shape[0]
                        # append only the one right before drug (i.e. max grown cells)
                        cell_line_encodings.append(encodings[n_times_before_drug - 1, :])
                        drug_names.append(well_names[i])

                    elif time_point == 'end':
                        # append only the last one (i.e. max drug effect)
                        cell_line_encodings.append(encodings[time.shape[0] - 1, :])
                        drug_names.append(well_names[i])

                    elif isinstance(time_point, int) or isinstance(time_point, float):
                        # TODO: implement retrieving a particular time point if necessary
                        pass
                    else:
                        raise ValueError("Time point not known!")

    try:
        assert len(drug_names) == len(cell_line_encodings)
    except AssertionError:
        print('drug names size: {}'.format(len(drug_names)))
        print('encodings size: {}'.format(len(cell_line_encodings)))

    cell_line_encodings = numpy.array(cell_line_encodings).astype('float32')

    print('single cell line dataset shape:', cell_line_encodings.shape)

    return cell_line_encodings, drug_names


def collect_encodings_by_time_point(n_batches_to_use, time_point='zero', path_to_batches="/Users/andreidm/ETH/projects/pheno-ml/data/"):

    cell_line_names = []
    cell_line_encodings = []

    for batch in range(1, n_batches_to_use+1):
        print('collecting from batch {}:'.format(batch))
        path_to_batch = path_to_batches + "batch_{}/".format(batch)
        for cell_line_folder in os.listdir(path_to_batch):
            print('collecting from folder {}...'.format(cell_line_folder))

            cell_line_name = cell_line_folder.split('_')[0]

            if cell_line_folder.startswith("."):
                continue
            else:
                path_to_encodings = path_to_batch + cell_line_folder + '/'

                # take only every second path to speed up calculations
                well_paths = [file for file in os.listdir(path_to_encodings) if file.endswith('.csv')][::2]

                for well in well_paths:

                    well_encodings = pandas.read_csv(path_to_encodings + well).values
                    time = well_encodings[:, 1].astype('float32')
                    encodings = well_encodings[:, 2:].astype('float32')

                    if time_point == 'zero':
                        # get number of encodings before drug administration
                        n_times_before_drug = time[time < 0].shape[0]
                        # append only the one right before drug (i.e. max grown cells)
                        cell_line_encodings.append(encodings[n_times_before_drug - 1, :])
                    elif time_point == 'end':
                        # append only the last one (i.e. max drug effect)
                        cell_line_encodings.append(encodings[time.shape[0] - 1, :])
                    elif isinstance(time_point, int) or isinstance(time_point, float):
                        # TODO: implement retrieving a particular time point if necessary
                        pass
                    else:
                        raise ValueError("Time point not known!")

                    # append name of the cell line
                    cell_line_names.append(cell_line_name)

    assert len(cell_line_names) == len(cell_line_encodings)
    cell_line_encodings = numpy.array(cell_line_encodings).astype('float32')

    print('the whole dataset shape:', cell_line_encodings.shape)

    return cell_line_encodings, cell_line_names


def test_umap_stability(neighbors=None, metric='euclidean', min_dist=0.1):
    """ This function runs UMAP on different parameters to explore the stability of the method. """

    save_plots_to = '/Users/andreidm/ETH/projects/pheno-ml/res/umap_embeddings/'

    if neighbors is None:
        neighbors = [15, 30, 60, 100, 200, 300]

    data, names = collect_encodings_by_time_point(1, time_point='zero')

    start = time.time()
    scaled_data = StandardScaler().fit_transform(data)
    print('scaling took {} s'.format(time.time() - start))

    pyplot.subplots(nrows=2, ncols=3, figsize=(12,6))
    seaborn.set(font_scale=0.5)
    seaborn.color_palette('colorblind')
    seaborn.axes_style('whitegrid')

    for i, n in enumerate(neighbors):

        reducer = umap.UMAP(n_neighbors=n, metric=metric, min_dist=min_dist)
        start = time.time()
        embedding = reducer.fit_transform(scaled_data)
        print('umap transform with n = {} took {} s'.format(n, time.time() - start))

        pyplot.subplot(2, 3, i + 1)
        seaborn.scatterplot(x=embedding[:, 0], y=embedding[:, 1], hue=names, alpha=0.5, s=15)
        pyplot.title('n={}, metric={}, min_dist={}'.format(n, metric, min_dist), fontsize=8)

    # pyplot.show()
    pyplot.savefig(save_plots_to + 'umap_batch1_{}.pdf'.format(metric))

    print("plot saved to", save_plots_to)


def perform_full_data_umap_and_plot_embeddings(time_point='zero', n=15, metric='euclidean', min_dist=0.1, save_to='/Users/andreidm/ETH/projects/pheno-ml/res/umap_embeddings/'):

    data, names = collect_encodings_by_time_point(7, time_point=time_point)

    start = time.time()
    scaled_data = StandardScaler().fit_transform(data)
    print('scaling took {} s'.format(time.time() - start))

    reducer = umap.UMAP()
    start = time.time()
    embedding = reducer.fit_transform(scaled_data)
    print('umap transform with n = {} took {} s'.format(n, time.time() - start))

    pyplot.figure()
    seaborn.set(font_scale=0.5)
    seaborn.color_palette('colorblind')
    seaborn.axes_style('whitegrid')
    seaborn.scatterplot(x=embedding[:, 0], y=embedding[:, 1], hue=names, alpha=0.6, s=10)
    pyplot.title('UMAP: n={}, metric={}, min_dist={}'.format(n, metric, min_dist), fontsize=8)

    pyplot.savefig(save_to + 'umap_time={}_n={}_metric={}.pdf'.format(time_point, n, metric))
    pyplot.close('all')
    print('plot saved')


def perform_umap_for_cell_line_and_plot_drug_effects(cell_line, time_point, n=5, metric='euclidean', min_dist=0.1, save_to='/Users/andreidm/ETH/projects/pheno-ml/res/umap_embeddings/cell_lines/'):

    data, names = collect_encodings_of_cell_line_by_time_points(cell_line, time_point=time_point)
    n_drugs = len(set(names))
    print('number of drugs: {}'.format(n_drugs))

    start = time.time()
    scaled_data = StandardScaler().fit_transform(data)
    print('scaling took {} s'.format(time.time() - start))

    seaborn.set(font_scale=1.)
    seaborn.color_palette('colorblind')
    seaborn.axes_style('whitegrid')

    reducer = umap.UMAP(n_neighbors=n, metric=metric, min_dist=min_dist)
    start = time.time()
    embedding = reducer.fit_transform(scaled_data)
    print('umap transform with n = {} took {} s'.format(n, time.time() - start))

    if n_drugs > 18:
        legend_font_size = 6
    else:
        legend_font_size = 10

    pyplot.figure()
    seaborn.scatterplot(x=embedding[:, 0], y=embedding[:, 1], hue=names, alpha=0.8, s=20)
    pyplot.title('cell line: {}, time point: {}'.format(cell_line, time_point))
    pyplot.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize=legend_font_size)
    pyplot.tight_layout()

    # pyplot.show()
    pyplot.savefig(save_to + 'umap_{}_{}.pdf'.format(cell_line, time_point))
    pyplot.close('all')
    print('plot saved')


if __name__ == '__main__':

    if False:
        n = [15, 30, 60, 100, 200, 300]
        test_umap_stability(neighbors=n, metric='correlation')
        test_umap_stability(neighbors=n, metric='braycurtis')
        test_umap_stability(neighbors=n, metric='cosine', min_dist=0.25)

        n = [5, 10, 15, 100, 150, 500]
        test_umap_stability(neighbors=n, metric='euclidean')

    # TODO: implement umap and plotting for single batches?

    if False:
        perform_full_data_umap_and_plot_embeddings(time_point='zero', n=50, metric='cosine')
        perform_full_data_umap_and_plot_embeddings(time_point='end', n=50, metric='cosine')

    if True:

        for cell_line in constants.cell_lines[2:]:
            print('\nperforming umap for {}...'.format(cell_line))
            perform_umap_for_cell_line_and_plot_drug_effects(cell_line, 'zero')
            perform_umap_for_cell_line_and_plot_drug_effects(cell_line, 'end')
