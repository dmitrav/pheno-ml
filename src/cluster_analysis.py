
import numpy, pandas, os, umap, seaborn, time
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot


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
                for well in os.listdir(path_to_encodings):
                    if well.endswith('.csv'):

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
    cell_line_encodings = numpy.array(cell_line_encodings)

    print('dataset shape:', cell_line_encodings.shape)

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


def perform_umap_and_plot_embeddings(time_point='zero', n=15, metric='euclidean', min_dist=0.1, save_to='/Users/andreidm/ETH/projects/pheno-ml/res/umap_embeddings/'):

    data, names = collect_encodings_by_time_point(7, time_point=time_point)

    start = time.time()
    scaled_data = StandardScaler().fit_transform(data)
    print('scaling took {} s'.format(time.time() - start))

    reducer = umap.UMAP()
    start = time.time()
    embedding = reducer.fit_transform(scaled_data)
    print('umap transform with n = {} took {} s'.format(n, time.time() - start))

    seaborn.set(font_scale=0.5)
    seaborn.color_palette('colorblind')
    seaborn.axes_style('whitegrid')
    seaborn.scatterplot(x=embedding[:, 0], y=embedding[:, 1], hue=names, alpha=0.5, s=15)
    pyplot.title('UMAP: n={}, metric={}, min_dist={}'.format(n, metric, min_dist), fontsize=8)

    pyplot.savefig(save_to + 'umap_time=\'{}\'.pdf'.format(time_point))
    print('plot saved')


if __name__ == '__main__':

    if True:
        n = [15, 30, 60, 100, 200, 300]
        test_umap_stability(neighbors=n, metric='correlation')
        test_umap_stability(neighbors=n, metric='braycurtis')
        test_umap_stability(neighbors=n, metric='cosine', min_dist=0.25)

        n = [5, 10, 15, 100, 150, 500]
        test_umap_stability(neighbors=n, metric='euclidean')

    if False:
        perform_umap_and_plot_embeddings(time_point='zero')
        perform_umap_and_plot_embeddings(time_point='end')

