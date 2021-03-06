
import numpy, pandas, os, umap, seaborn, time
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
from src import constants
from tqdm import tqdm


def get_well_drug_mapping_for_cell_line(cell_line_folder, path_to_meta='/Users/andreidm/ETH/projects/pheno-ml/data/metadata/'):
    """ Retrieve a well-to-drug mapping for a cell line folder. """

    cell_line_meta = pandas.read_csv(path_to_meta + cell_line_folder + '.csv')
    wells = cell_line_meta['Well'].values
    drugs = cell_line_meta['Drug'].values
    concs = cell_line_meta['Final_conc_uM'].values
    del cell_line_meta

    mapping = {}
    for i in range(wells.shape[0]):
        mapping[wells[i]] = (drugs[i], concs[i])

    return mapping


def get_wells_of_single_drug_for_cell_line(drug, path_to_meta='/Users/andreidm/ETH/projects/pheno-ml/data/pheno-ml-metadata.csv'):

    meta = pandas.read_csv(path_to_meta)

    max_concentration = meta.loc[meta['Drug'] == drug, 'Final_conc_uM'].max()
    wells = meta.loc[(meta['Drug'] == drug) & (meta['Final_conc_uM'] == max_concentration), 'Well'].unique()

    return wells, max_concentration


def collect_encodings_of_cell_line_by_time_points(cell_line_name, time_point='zero', path_to_batches="/Users/andreidm/ETH/projects/pheno-ml/data/cropped/"):

    drug_names = []
    drug_concs = []
    cell_line_encodings = []
    exact_time_points = []
    image_ids = []  # plate + well

    for batch in range(1, 8):
        path_to_batch = path_to_batches + "batch_{}/".format(batch)
        for folder in os.listdir(path_to_batch):
            if cell_line_name in folder:

                well_drug_map = get_well_drug_mapping_for_cell_line(folder)

                path_to_encodings = path_to_batch + folder + '/'

                well_paths = [file for file in os.listdir(path_to_encodings) if file.endswith('.csv')]
                well_drugs = [well_drug_map[file.replace('.csv', '')][0] for file in os.listdir(path_to_encodings) if file.endswith('.csv')]
                well_concs = [well_drug_map[file.replace('.csv', '')][1] for file in os.listdir(path_to_encodings) if file.endswith('.csv')]

                for i in range(len(well_paths)):

                    well_id = folder + '_' + well_paths[i].split('.')[0]

                    well_encodings = pandas.read_csv(path_to_encodings + well_paths[i]).values
                    time = well_encodings[:, 1].astype('float32')
                    encodings = well_encodings[:, 2:].astype('float32')
                    del well_encodings

                    if time_point == 'zero':
                        # get number of encodings before drug administration
                        n_times_before_drug = time[time < 0].shape[0]
                        # append only the one right before drug (i.e. max grown cells)
                        cell_line_encodings.append(encodings[n_times_before_drug - 1, :])
                        drug_names.append(well_drugs[i])
                        drug_concs.append(well_concs[i])
                        image_ids.append(well_id)
                        exact_time_points.append(numpy.max(time[time < 0]))

                    elif time_point == 'end':
                        # append only the last one (i.e. max drug effect)
                        cell_line_encodings.append(encodings[time.shape[0] - 1, :])
                        drug_names.append(well_drugs[i])
                        drug_concs.append(well_concs[i])
                        image_ids.append(well_id)
                        exact_time_points.append(time[-1])

                    elif time_point == 'all':
                        cell_line_encodings.extend(encodings)
                        drug_names.extend([well_drugs[i] for x in range(encodings.shape[0])])
                        drug_concs.extend([well_concs[i] for x in range(encodings.shape[0])])
                        image_ids.extend([well_id for x in range(encodings.shape[0])])
                        exact_time_points.extend(time)

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

    return cell_line_encodings, drug_names, drug_concs, image_ids, exact_time_points


def collect_encodings_of_drug_by_time_points(drug, time_point='zero', path_to_batches="/Users/andreidm/ETH/projects/pheno-ml/data/cropped/"):

    cell_line_names = []
    drug_encodings = []
    drug_concs = []
    exact_time_points = []
    image_ids = []  # plate + well

    for batch in range(1, 8):
        path_to_batch = path_to_batches + "batch_{}/".format(batch)
        for cell_line_folder in os.listdir(path_to_batch):

            if cell_line_folder != '.DS_Store':
                path_to_encodings = path_to_batch + cell_line_folder + '/'

                well_names, max_conc = get_wells_of_single_drug_for_cell_line(drug)  # 4 replicates
                well_paths = [name + '.csv' for name in well_names]

                for i in range(len(well_paths)):

                    well_id = cell_line_folder + '_' + well_names[i]

                    try:
                        well_encodings = pandas.read_csv(path_to_encodings + well_paths[i]).values
                        time = well_encodings[:, 1].astype('float32')
                        encodings = well_encodings[:, 2:].astype('float32')
                        del well_encodings

                        if time_point == 'zero':
                            # get number of encodings before drug administration
                            n_times_before_drug = time[time < 0].shape[0]
                            # append only the one right before drug (i.e. max grown cells)
                            drug_encodings.append(encodings[n_times_before_drug - 1, :])
                            cell_line_names.append(cell_line_folder.split('_')[0])
                            image_ids.append(well_id)
                            exact_time_points.append(numpy.max(time[time < 0]))
                            drug_concs.append(max_conc)

                        elif time_point == 'end':
                            # append only the last one (i.e. max drug effect)
                            drug_encodings.append(encodings[time.shape[0] - 1, :])
                            cell_line_names.append(cell_line_folder.split('_')[0])
                            image_ids.append(well_id)
                            exact_time_points.append(time[-1])
                            drug_concs.append(max_conc)

                        elif time_point == 'all':
                            drug_encodings.extend(encodings)
                            cell_line_names.extend([cell_line_folder.split('_')[0] for x in range(encodings.shape[0])])
                            image_ids.extend([well_id for x in range(encodings.shape[0])])
                            drug_concs.extend([max_conc for x in range(encodings.shape[0])])
                            exact_time_points.extend(time)

                        elif isinstance(time_point, int) or isinstance(time_point, float):
                            # TODO: implement retrieving a particular time point if necessary
                            pass
                        else:
                            raise ValueError("Time point not known!")

                    except FileNotFoundError:
                        print('Drug {}: file {} not found in folder {}...'.format(drug, well_paths[i], cell_line_folder))
    try:
        assert len(cell_line_names) == len(drug_encodings)
    except AssertionError:
        print('Assertion error!')
        print('cell line names size: {}'.format(len(cell_line_names)))
        print('drug encodings size: {}'.format(len(drug_encodings)))

    drug_encodings = numpy.array(drug_encodings).astype('float32')

    print('single drug dataset shape:', drug_encodings.shape)

    return drug_encodings, cell_line_names, drug_concs, image_ids, exact_time_points


def collect_encodings_by_time_point(n_batches_to_use, time_point='zero', path_to_batches="/Users/andreidm/ETH/projects/pheno-ml/data/cropped/"):

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


def test_umap_stability(neighbors=None, metrics=['euclidean'], min_dist=0.1):
    """ This function runs UMAP on different parameters to explore the stability of the method. """

    save_plots_to = '/Users/andreidm/ETH/projects/pheno-ml/res/umap_embeddings/consistency_check/'

    if neighbors is None:
        neighbors = [15, 30, 60, 100, 200, 300]
    elif len(neighbors) > 6:
        neighbors = neighbors[:6]

    data, names = collect_encodings_by_time_point(1, time_point='zero')

    start = time.time()
    scaled_data = StandardScaler().fit_transform(data)
    print('scaling took {} s'.format(time.time() - start))

    for metric in metrics:

        pyplot.figure()
        pyplot.subplots(nrows=2, ncols=3, figsize=(12, 6))
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


def perform_full_data_umap_and_plot_embeddings(time_point='zero', parameters=[(15, 'euclidean', 0.1)], save_to='/Users/andreidm/ETH/projects/pheno-ml/res/umap_embeddings/'):

    data, names = collect_encodings_by_time_point(7, time_point=time_point)

    start = time.time()
    scaled_data = StandardScaler().fit_transform(data)
    print('scaling took {} s'.format(time.time() - start))

    for triple in parameters:

        neighbors, metric, min_dist = triple

        reducer = umap.UMAP(n_neighbors=neighbors, metric=metric, min_dist=min_dist)
        start = time.time()
        embedding = reducer.fit_transform(scaled_data)
        print('umap transform with n = {} took {} s'.format(neighbors, time.time() - start))

        pyplot.figure()
        seaborn.set(font_scale=0.5)
        seaborn.color_palette('colorblind')
        seaborn.axes_style('whitegrid')
        seaborn.scatterplot(x=embedding[:, 0], y=embedding[:, 1], hue=names, s=10)
        pyplot.title('UMAP: n={}, metric={}, min_dist={}'.format(neighbors, metric, min_dist), fontsize=8)

        pyplot.savefig(save_to + 'umap_cropped_time={}_n={}_metric={}.pdf'.format(time_point, neighbors, metric))
        pyplot.close('all')
        print('plot saved')


def perform_umap_for_cell_line_and_plot_drug_effects(cell_line, time_point, n=15, metric='euclidean', min_dist=0.1, annotate_points=False, save_to='/Users/andreidm/ETH/projects/pheno-ml/res/umap_embeddings/cell_lines/'):

    data, drug_names, drug_concs, image_ids, exact_tps = collect_encodings_of_cell_line_by_time_points(cell_line, time_point=time_point)
    unique_drug_names = list(set(drug_names))
    print('number of drugs: {}'.format(len(unique_drug_names)))

    df = pandas.DataFrame({'drug': drug_names, 'conc': drug_concs, 'wells': image_ids, 'time': exact_tps})
    df = pandas.concat([df, pandas.DataFrame(data)], axis=1)
    df.to_csv(save_to + cell_line + ".csv", index=False)
    del df

    start = time.time()
    scaled_data = StandardScaler().fit_transform(data)
    print('scaling took {} s'.format(time.time() - start))

    seaborn.set(font_scale=1.)
    seaborn.color_palette('colorblind')
    seaborn.axes_style('whitegrid')

    reducer = umap.UMAP(n_neighbors=n, metric=metric, min_dist=min_dist, random_state=27)
    start = time.time()
    embedding = reducer.fit_transform(scaled_data)
    print('umap transform with n = {} took {} s'.format(n, time.time() - start))

    if len(unique_drug_names) > 18:
        legend_font_size = 6
    else:
        legend_font_size = 10

    fig = pyplot.figure()
    ax = fig.add_subplot(111, projection='3d')

    colors = [unique_drug_names.index(x) for x in drug_names]
    ax.scatter(embedding[:, 0], exact_tps, embedding[:, 1], c=colors, s=15, cmap='rainbow', edgecolors='black', alpha=0.7, linewidth=.5)

    ax.set_ylabel('Time (hours)')

    pyplot.title('cell line: {}, time point: {}'.format(cell_line, time_point))
    pyplot.legend()
    pyplot.tight_layout()

    # pyplot.show()
    pyplot.savefig(save_to + 'umap_cropped_{}_{}_3d.pdf'.format(cell_line, time_point))


    pyplot.figure()
    seaborn.scatterplot(x=embedding[:, 0], y=embedding[:, 1], hue=drug_names, alpha=1., s=15)
    pyplot.title('cell line: {}, time point: {}'.format(cell_line, time_point))
    pyplot.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize=legend_font_size)
    pyplot.tight_layout()
    # pyplot.show()
    pyplot.savefig(save_to + 'umap_cropped_{}_{}_drugs.pdf'.format(cell_line, time_point))

    if annotate_points:
        for i in range(len(image_ids)):
            pyplot.annotate(image_ids[i],  # this is the text
                            (embedding[i, 0], embedding[i, 1]),  # this is the point to label
                            textcoords="offset points",  # how to position the text
                            xytext=(0, 3),  # distance from text to points (x,y)
                            ha='center',  # horizontal alignment can be left, right or center
                            fontsize=6)

    cmap = seaborn.cubehelix_palette(as_cmap=True)
    fig, ax = pyplot.subplots()

    p = ax.scatter(x=embedding[:, 0], y=embedding[:, 1], c=exact_tps, s=15, cmap=cmap, edgecolors='black', linewidth=.5)
    fig.colorbar(p, ax=ax, label='Time (hours)')

    pyplot.title('cell line: {}, time point: {}'.format(cell_line, time_point))
    pyplot.legend()
    pyplot.tight_layout()

    # pyplot.show()
    pyplot.savefig(save_to + 'umap_cropped_{}_{}_times.pdf'.format(cell_line, time_point))
    pyplot.close('all')
    print('plot saved')


def perform_umap_for_drug_and_plot_results(drug, time_point='end', n=15, metric='braycurtis', min_dist=0.1, annotate_points=False, save_to='/Users/andreidm/ETH/projects/pheno-ml/res/umap_embeddings/drugs/'):

    data, cell_names, drug_concs, image_ids, exact_tps = collect_encodings_of_drug_by_time_points(drug, time_point=time_point)

    unique_cell_names = list(set(cell_names))
    assert len(unique_cell_names) == len(constants.cell_lines)
    print('number of cell lines collected: {}'.format(len(unique_cell_names)))

    df = pandas.DataFrame({'drug': cell_names, 'conc': drug_concs, 'wells': image_ids, 'time': exact_tps})
    df = pandas.concat([df, pandas.DataFrame(data)], axis=1)
    df.to_csv(save_to + drug + ".csv", index=False)
    del df

    start = time.time()
    scaled_data = StandardScaler().fit_transform(data)
    print('scaling took {} s'.format(time.time() - start))

    seaborn.set(font_scale=1.)
    seaborn.color_palette('colorblind')
    seaborn.axes_style('whitegrid')

    reducer = umap.UMAP(n_neighbors=n, metric=metric, min_dist=min_dist, random_state=29)
    start = time.time()
    embedding = reducer.fit_transform(scaled_data)
    print('umap transform with n = {} took {} s'.format(n, time.time() - start))

    fig = pyplot.figure()
    ax = fig.add_subplot(111, projection='3d')

    colors = [unique_cell_names.index(x) for x in cell_names]
    ax.scatter(exact_tps, embedding[:, 0], embedding[:, 1], c=colors, s=15, cmap='rainbow', edgecolors='black',
               alpha=0.7, linewidth=.5)

    ax.set_xlabel('Time (hours)')

    pyplot.title('drug: {}, time point: {}'.format(drug, time_point))
    pyplot.legend()
    pyplot.tight_layout()

    # pyplot.show()
    pyplot.savefig(save_to + 'umap_cropped_{}_{}_3d.pdf'.format(drug, time_point))

    pyplot.figure()
    seaborn.scatterplot(x=embedding[:, 0], y=embedding[:, 1], hue=cell_names, alpha=1., s=20)

    if annotate_points:
        for i in range(len(image_ids)):
            pyplot.annotate(image_ids[i],  # this is the text
                            (embedding[i, 0], embedding[i, 1]),  # this is the point to label
                            textcoords="offset points",  # how to position the text
                            xytext=(0, 3),  # distance from text to points (x,y)
                            ha='center',  # horizontal alignment can be left, right or center
                            fontsize=6)

    pyplot.title('drug: {}, time point: {}'.format(drug, time_point))
    pyplot.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize=6)
    pyplot.tight_layout()

    # pyplot.show()
    pyplot.savefig(save_to + 'umap_cropped_{}_{}_cells.pdf'.format(drug, time_point))

    cmap = seaborn.cubehelix_palette(as_cmap=True)
    fig, ax = pyplot.subplots()

    p = ax.scatter(x=embedding[:, 0], y=embedding[:, 1], c=exact_tps, s=15, cmap=cmap, edgecolors='black', linewidth=.5)
    fig.colorbar(p, ax=ax, label='Time (hours)')

    pyplot.title('drug: {}, time point: {}'.format(drug, time_point))
    pyplot.legend()
    pyplot.tight_layout()
    # pyplot.show()
    pyplot.savefig(save_to + 'umap_cropped_{}_{}_times.pdf'.format(drug, time_point))

    pyplot.close('all')
    print('plot saved')


if __name__ == '__main__':

    if False:
        n = [5, 15, 50, 100, 300, 500]
        metrics = ['correlation', 'braycurtis', 'cosine', 'euclidean']

        test_umap_stability(neighbors=n, metrics=metrics)

    if False:

        parameters = [
            (15, 'euclidean', 0.1)
        ]

        perform_full_data_umap_and_plot_embeddings(time_point='zero', parameters=parameters, save_to='/Users/andreidm/ETH/projects/pheno-ml/res/umap_embeddings/whole_dataset/cropped/')
        perform_full_data_umap_and_plot_embeddings(time_point='end', parameters=parameters, save_to='/Users/andreidm/ETH/projects/pheno-ml/res/umap_embeddings/whole_dataset/cropped/')

    if False:

        # for cell_line in tqdm(constants.cell_lines):
        for cell_line in ['ACHN']:
            print('\nperforming umap for {}...'.format(cell_line))
            # perform_umap_for_cell_line_and_plot_drug_effects(cell_line, time_point='zero', n=15, metric='euclidean')
            perform_umap_for_cell_line_and_plot_drug_effects(cell_line, time_point='end', n=15, metric='euclidean', annotate_points=True)

    if True:

        for cell_line in tqdm(constants.cell_lines):
        # for cell_line in ['ACHN']:
            print('\nperforming umap for {}...'.format(cell_line))
            perform_umap_for_cell_line_and_plot_drug_effects(cell_line, time_point='all', n=15, metric='euclidean', min_dist=1.,
                                                             annotate_points=False, save_to='/Users/andreidm/ETH/projects/pheno-ml/res/embeddings/')

    if False:
        # for drug in tqdm(constants.drugs):
        for drug in ['Cladribine']:
            print('\nperforming umap for {}...'.format(drug))
            # perform_umap_for_drug_and_plot_results(drug, time_point='zero', n=5, metric='euclidean')
            # only max concentrations are used there
            perform_umap_for_drug_and_plot_results(drug, time_point='end', n=5, metric='euclidean', annotate_points=True)

    if False:
        for drug in tqdm(constants.drugs):
        # for drug in ['Cladribine']:
            print('\nperforming umap for {}...'.format(drug))
            # only max concentrations are used there
            perform_umap_for_drug_and_plot_results(drug, time_point='all', n=5, metric='euclidean', annotate_points=False,
                                                   save_to='/Users/andreidm/ETH/projects/pheno-ml/res/embeddings/drugs/')



