
import numpy, pandas, os, umap, seaborn, time, hdbscan

from src.constants import user, cropped_data_path
from src.constants import drugs as all_drug_names


from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import fcluster, linkage
from matplotlib import pyplot
from sklearn.metrics import silhouette_score
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm
from datetime import datetime
from shutil import copyfile


def get_well_drug_mapping_for_cell_line(cell_line_folder, keep_max_conc_only=False, path_to_meta='/Users/{}/ETH/projects/pheno-ml/data/metadata/'.format(user)):
    """ Retrieve a well-to-drug mapping for a cell line folder. """

    cell_line_meta = pandas.read_csv(path_to_meta + cell_line_folder + '.csv')
    wells = cell_line_meta['Well'].values
    drugs = cell_line_meta['Drug'].values
    concs = cell_line_meta['Final_conc_uM'].values
    del cell_line_meta

    if keep_max_conc_only:

        i = 0
        while i < len(drugs):

            if drugs[i] in [*all_drug_names, 'DMSO']:
                # find max conc of the drug
                drug_max_conc = max(concs[drugs == drugs[i]])
                # remove a "column", if it's not related to max conc
                if concs[i] != drug_max_conc:
                    wells = numpy.delete(wells, i)
                    drugs = numpy.delete(drugs, i)
                    concs = numpy.delete(concs, i)
                else:
                    i += 1
            else:
                i += 1

    mapping = {}
    for i in range(wells.shape[0]):
        mapping[wells[i]] = (drugs[i], concs[i])

    return mapping


def get_wells_of_single_drug_for_cell_line(drug, path_to_meta='/Users/{}/ETH/projects/pheno-ml/data/pheno-ml-metadata.csv'.format(user)):

    meta = pandas.read_csv(path_to_meta)

    max_concentration = meta.loc[meta['Drug'] == drug, 'Final_conc_uM'].max()
    wells = meta.loc[(meta['Drug'] == drug) & (meta['Final_conc_uM'] == max_concentration), 'Well'].unique()

    return wells, max_concentration


def collect_encodings_of_cell_line_by_time_points(cell_line_name, time_point='zero', keep_max_conc_only=False, path_to_batches="/Users/{}/ETH/projects/pheno-ml/data/cropped/".format(user)):

    drug_names = []
    drug_concs = []
    cell_line_encodings = []
    image_dates = []
    exact_time_points = []
    image_ids = []  # plate + well

    for batch in range(1, 8):
        path_to_batch = path_to_batches + "batch_{}/".format(batch)
        for folder in os.listdir(path_to_batch):
            if cell_line_name in folder:

                well_drug_map = get_well_drug_mapping_for_cell_line(folder, keep_max_conc_only=keep_max_conc_only)

                path_to_encodings = path_to_batch + folder + '/'

                well_paths = [file for file in os.listdir(path_to_encodings) if file.endswith('.csv') and file.replace('.csv', '') in well_drug_map.keys()]
                well_drugs = [well_drug_map[file.replace('.csv', '')][0] for file in os.listdir(path_to_encodings) if file.endswith('.csv') and file.replace('.csv', '') in well_drug_map.keys()]
                well_concs = [well_drug_map[file.replace('.csv', '')][1] for file in os.listdir(path_to_encodings) if file.endswith('.csv') and file.replace('.csv', '') in well_drug_map.keys()]

                for i in range(len(well_paths)):

                    well_id = folder + '_' + well_paths[i].split('.')[0]

                    well_encodings = pandas.read_csv(path_to_encodings + well_paths[i]).values
                    well_dates = well_encodings[:, 0]
                    time = well_encodings[:, 1].astype('float32')
                    encodings = well_encodings[:, 2:].astype('float32')
                    del well_encodings

                    if time_point == 'zero':
                        # get number of encodings before drug administration
                        n_times_before_drug = time[time < 0].shape[0]
                        # append only the one right before drug (i.e. max grown cells)
                        cell_line_encodings.append(encodings[n_times_before_drug - 1, :])
                        image_dates.append(well_dates[n_times_before_drug - 1])
                        drug_names.append(well_drugs[i])
                        drug_concs.append(well_concs[i])
                        image_ids.append(well_id)
                        exact_time_points.append(numpy.max(time[time < 0]))

                    elif time_point == 'end':
                        # append only the last one (i.e. max drug effect)
                        cell_line_encodings.append(encodings[time.shape[0] - 1, :])
                        image_dates.append(well_dates[time.shape[0] - 1])
                        drug_names.append(well_drugs[i])
                        drug_concs.append(well_concs[i])
                        image_ids.append(well_id)
                        exact_time_points.append(time[-1])

                    elif time_point == 'all':
                        cell_line_encodings.extend(encodings)
                        image_dates.extend(well_dates)
                        drug_names.extend([well_drugs[i] for x in range(encodings.shape[0])])
                        drug_concs.extend([well_concs[i] for x in range(encodings.shape[0])])
                        image_ids.extend([well_id for x in range(encodings.shape[0])])
                        exact_time_points.extend(time)

                    elif time_point == 'last10':
                        cell_line_encodings.extend(encodings[-10:, :])
                        image_dates.extend(well_dates[-10:])
                        drug_names.extend([well_drugs[i] for x in range(encodings.shape[0]-10, encodings.shape[0])])
                        drug_concs.extend([well_concs[i] for x in range(encodings.shape[0]-10, encodings.shape[0])])
                        image_ids.extend([well_id for x in range(encodings.shape[0]-10, encodings.shape[0])])
                        exact_time_points.extend(time[-10:])

                    elif isinstance(time_point, int) or isinstance(time_point, float):
                        # implement retrieving a particular time point if necessary
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

    return cell_line_encodings, drug_names, drug_concs, image_ids, exact_time_points, image_dates


def collect_encodings_of_drug_by_time_points(drug, time_point='zero', path_to_batches="/Users/{}/ETH/projects/pheno-ml/data/cropped/".format(user)):

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


def collect_encodings_by_time_point(n_batches_to_use, time_point='zero', path_to_batches="/Users/{}/ETH/projects/pheno-ml/data/cropped/".format(user)):

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

    save_plots_to = '/Users/{}/ETH/projects/pheno-ml/res/umap_embeddings/consistency_check/'.format(user)

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


def perform_full_data_umap_and_plot_embeddings(time_point='zero', parameters=[(15, 'euclidean', 0.1)], save_to='/Users/{}/ETH/projects/pheno-ml/res/umap_embeddings/'.format(user)):

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


def perform_umap_for_cell_line_and_plot_drug_effects(cell_line, time_point, n=15, metric='euclidean', min_dist=0.1, annotate_points=False, save_to='/Users/{}/ETH/projects/pheno-ml/res/umap_embeddings/cell_lines/'.format(user)):

    data, drug_names, drug_concs, image_ids, exact_tps, _ = collect_encodings_of_cell_line_by_time_points(cell_line, time_point=time_point)
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


def save_clustered_image_examples(clustering, image_ids, image_dates, drug_names, cell_line, save_to, N=10):
    """ This method saves N random examples of each cluster """

    clusters = numpy.unique(clustering.labels_)
    for cluster in clusters:

        if not os.path.exists(save_to + '{}/{}'.format(cell_line, cluster)):
            os.makedirs(save_to + '{}/{}'.format(cell_line, cluster))

        cluster_indices = numpy.where(clustering.labels_ == cluster)[0]
        images_to_save = numpy.array(image_ids)[cluster_indices]
        dates_to_save = numpy.array(image_dates)[cluster_indices]
        drugs_to_save = numpy.array(drug_names)[cluster_indices]

        if images_to_save.shape[0] > N:
            random_indices = numpy.random.choice([x for x in range(len(images_to_save))], size=N, replace=False)
            images_to_save = images_to_save[random_indices]
            dates_to_save = dates_to_save[random_indices]
            drugs_to_save = drugs_to_save[random_indices]

        for i, image in enumerate(images_to_save):
            image_folder = '_'.join(image.split('_')[:-1])
            image_key = image.split('_')[-1]

            # find related images
            for batch in range(1,8):

                batch_path = cropped_data_path + 'batch_{}/'.format(batch)
                if image_folder in os.listdir(batch_path):

                    date_obj = datetime.strptime(dates_to_save[i], '%Y-%m-%d %H:%M:%S')
                    image_date = datetime.strftime(date_obj, '%Yy%mm%dd_%Hh%Mm')

                    sought_image = [x for x in os.listdir(batch_path+image_folder) if image_key in x and image_date in x][0]

                    # save this image
                    copyfile(batch_path + image_folder + '/{}'.format(sought_image),
                             save_to + '{}/{}/{}_{}'.format(cell_line, cluster, drugs_to_save[i], sought_image))

                    break

                else:
                    continue


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

    if False:
        # for cell_line in tqdm(constants.cell_lines):
        for cell_line in ['ACHN']:
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
    if True:

        SEED = 4
        numpy.random.seed(SEED)
        save_images_to = '/Users/{}/ETH/projects/pheno-ml/res/clustering_within_cell_lines/'.format(user)

        # for cell_line in tqdm(constants.cell_lines):
        for cell_line in ['ACHN']:

            print('\nResults for {}:'.format(cell_line))

            data, drug_names, drug_concs, image_ids, exact_tps, dates = collect_encodings_of_cell_line_by_time_points(
                cell_line, time_point='last10', keep_max_conc_only=True)  # there's only cytotoxic effect at max conc

            unique_drug_names = list(set(drug_names))
            print('number of drugs: {}'.format(len(unique_drug_names)))

            df = pandas.DataFrame({'drug': drug_names, 'conc': drug_concs, 'wells': image_ids, 'time': exact_tps})
            df = pandas.concat([df, pandas.DataFrame(data)], axis=1)

            # set min cluster size
            any_drug = [x for x in df['drug'].unique() if x != 'DMSO'][0]
            min_cluster_size = df.loc[df['drug'] == any_drug, :].shape[0]
            reduced_dims = df.shape[1] / 10  # to get a matrix of around (2000, 400)

            reducer = umap.UMAP(n_components=reduced_dims, metric='euclidean', n_neighbors=min_cluster_size, min_dist=0.1, random_state=SEED)
            embeddings = reducer.fit_transform(df.iloc[:, 4:].values)

            # HDBSCAN
            clusterer = hdbscan.HDBSCAN(metric='euclidean', min_samples=1, min_cluster_size=min_cluster_size, allow_single_cluster=False)
            clusterer.fit(embeddings)

            total = clusterer.labels_.max() + 1
            print('n clusters={}'.format(total))
            save_clustered_image_examples(clusterer, image_ids, dates, drug_names, cell_line, save_images_to)

            # TODO:
            #  - plot heatmaps with clustering results
            #  - plot how many images are in each cluster



